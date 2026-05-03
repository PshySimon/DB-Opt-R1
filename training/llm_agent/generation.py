"""
Tool generation manager for LLM agents
"""

import torch
import numpy as np
import re
import json
import time
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import random

from .tensor_helper import TensorHelper, TensorConfig
from training.tool.tool_env import ToolEnv, step, step_batch
from training.progress import progress_heartbeat, progress_log
from verl import DataProto
from verl.utils.tracking import Tracking

@dataclass
class ToolGenerationConfig:
    """Configuration for tool-based generation"""
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_tool_response_length: int  # Renamed from max_obs_length
    num_gpus: int
    # use_parallel_tool_calls: bool = False
    use_batch_tool_calls: bool = False  # New option for batch execution
    tool_call_start: str = "<tool_call>"
    tool_call_end: str = "</tool_call>"
    tool_response_start: str = "<tool_response>"
    tool_response_end: str = "</tool_response>"
    tool_custom_response_template: str = ""
    strip_think_history: bool = True
    raw_prompt_history_turns: int = 4
    
class ToolGenerationManager:
    """Manager for handling LLM tool-based generation and interaction"""
    
    def __init__(
        self,
        tokenizer,
        sequence_generator,
        config: ToolGenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.sequence_generator = sequence_generator
        self.config = config
        self.is_validation = is_validation
        
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_tool_response_length=config.max_tool_response_length,  # Renamed
            max_start_length=config.max_start_length,
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _process_tool_call(self, responses_str) -> Tuple[List[str], List[bool]]:
        """
        Process a list of response strings to extract the first tool call
        while preserving the rest of the string content.
        
        Args:
            responses_str (List[str]): List of response strings potentially containing tool calls
            
        Returns:
            List[str]: Processed responses with only first tool call preserved
        """
        def process_single_response(resp):
            # Look for tool call pattern: <tool_call>tool_name(args)</tool_call>
            tool_pattern = r'<tool_call>(.*?)</tool_call>'
            match = re.search(tool_pattern, resp, re.DOTALL)
            
            if not match:
                return resp + self.tokenizer.eos_token, False  # No tool call found
            
            resp = resp.split(self.config.tool_call_end)[0] + self.config.tool_call_end
            # tool_content = match.group(0)
            
            # Replace all subsequent answer tag pairs with their content
            # rest_of_string = resp[match.end():]
            # cleaned_rest = re.sub(r'<tool_call>(.*?)</tool_call>', r'\1', rest_of_string, flags=re.DOTALL)
            
            return resp + self.tokenizer.eos_token, True
        
        # Process each response string
        return [process_single_response(resp)[0] for resp in responses_str], [process_single_response(resp)[1] for resp in responses_str]

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to extract tool calls."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # Extract the first tool call from each response
        responses_str, active_masks = self._process_tool_call(responses_str)
        
        # Tokenize processed responses
        cleaned_token_ids = self._batch_tokenize(responses_str)
        
        return cleaned_token_ids, responses_str, torch.tensor(active_masks, dtype=torch.bool)
    
    def _process_tool_responses(self, tool_responses: List[str]) -> torch.Tensor:
        """Process tool responses to token ids"""
        
        tool_responses_ids = self.tokenizer(
            tool_responses, 
            padding='longest',
            return_tensors='pt'
        )['input_ids']
        
        if tool_responses_ids.shape[1] > self.config.max_tool_response_length:
            print("[WARNING] TOOL RESPONSE TOO LONG, CONSIDER CHANGING YOUR CONFIG")
            tool_responses_ids = tool_responses_ids[:, :self.config.max_tool_response_length]
            
        return tool_responses_ids

    def _strip_think_history(self, response: str) -> str:
        return re.sub(r"<think>.*?</think>\s*", "", response or "", flags=re.DOTALL).strip()

    def _response_for_raw_prompt_history(self, response: str) -> str:
        response = self._strip_think_history(response)
        eos_token = getattr(self.tokenizer, "eos_token", None)
        if eos_token:
            response = response.replace(eos_token, "")
        return response.strip()

    def _responses_for_next_turn(self, responses_str: List[str]) -> torch.Tensor:
        if self.config.strip_think_history:
            responses_str = [self._strip_think_history(response) for response in responses_str]
        return self._batch_tokenize(responses_str)

    def _tool_response_message_content(self, tool_response: str) -> str:
        return (
            f"{self.config.tool_response_start}\n"
            f"{(tool_response or '').strip()}\n"
            f"{self.config.tool_response_end}"
        )

    def _trim_raw_prompt_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        max_history_turns = self.config.raw_prompt_history_turns
        if max_history_turns is None or max_history_turns < 0:
            return messages

        system_messages = []
        rest = messages
        if messages and messages[0].get("role") == "system":
            system_messages = [messages[0]]
            rest = messages[1:]

        current_user = []
        history = rest
        if rest and rest[-1].get("role") == "user":
            current_user = [rest[-1]]
            history = rest[:-1]

        keep_history = history[-2 * max_history_turns:] if max_history_turns else []
        if keep_history and keep_history[0].get("role") != "user":
            keep_history = keep_history[1:]

        return system_messages + keep_history + current_user

    def _update_raw_prompts_for_next_turn(
        self,
        rollings: DataProto,
        responses_str: List[str],
        raw_tool_responses: List[str],
        continue_mask: torch.Tensor,
    ) -> None:
        if "raw_prompt" not in rollings.non_tensor_batch:
            return

        continue_list = continue_mask.tolist() if isinstance(continue_mask, torch.Tensor) else list(continue_mask)
        raw_prompts = list(rollings.non_tensor_batch["raw_prompt"])
        for i, should_continue in enumerate(continue_list):
            if not should_continue:
                continue
            messages = [dict(message) for message in raw_prompts[i]]
            messages.append(
                {
                    "role": "assistant",
                    "content": self._response_for_raw_prompt_history(responses_str[i]),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": self._tool_response_message_content(raw_tool_responses[i]),
                }
            )
            raw_prompts[i] = self._trim_raw_prompt_history(messages)

        rollings.non_tensor_batch["raw_prompt"] = np.array(raw_prompts, dtype=object)
    
    def _execute_tool_calls(self, response_strs: List[str], 
                          envs: List[ToolEnv], 
                          active_mask: torch.Tensor) -> Tuple[List[str], List[bool]]:
        """Execute tool calls sequentially and return tool responses plus continue mask."""
        # Convert torch tensor to list of booleans if needed
        active_list = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
        
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs)
        raw_tool_responses = [""] * len(response_strs)
        continue_mask = [False] * len(response_strs)
        # Process each environment sequentially
        for i, (resp, env, active) in enumerate(zip(response_strs, envs, active_list)):
            if not active:
                continue
                
            # Step the environment using the agent's response
            result = step(env, resp)
            tool_response, _, done, _ = result
            raw_tool_responses[i] = tool_response
            tool_responses[i] = self.config.tool_custom_response_template.format(tool_response=tool_response)            
            continue_mask[i] = not done
        return tool_responses, continue_mask, raw_tool_responses
    
    def _execute_tool_calls_batch(self, response_strs: List[str], 
                                 envs: List[ToolEnv], 
                                 active_mask: torch.Tensor) -> Tuple[List[str], List[bool]]:
        """Execute tool calls in batch for tools that support batch operations."""
        # Convert torch tensor to list of booleans
        active_list = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
        
        # Filter active environments and responses
        active_envs = []
        active_responses = []
        active_indices = []
        
        for i, (env, resp, active) in enumerate(zip(envs, response_strs, active_list)):
            if active:
                active_envs.append(env)
                active_responses.append(resp)
                active_indices.append(i)
        
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs)
        raw_tool_responses = [""] * len(response_strs)
        continue_mask = [False] * len(response_strs)
        
        if not active_envs:
            return tool_responses, continue_mask, raw_tool_responses
            
        # Use the independent step_batch function for active environments
        batch_results = step_batch(active_envs, active_responses)
        
        # Map results back to original indices
        for idx, result in zip(active_indices, batch_results):
            if result is None:
                tool_responses[idx] = ""
            else:
                tool_response, _, done, _ = result
                raw_tool_responses[idx] = tool_response
                tool_responses[idx] = self.config.tool_custom_response_template.format(tool_response=tool_response)
                continue_mask[idx] = not done
        return tool_responses, continue_mask, raw_tool_responses
    
    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            tool_responses_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            tool_responses_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return DataProto.from_dict(
            tensors={
                'input_ids': new_input_ids[:, -max_len:],
                'position_ids': new_position_ids[:, -max_len:],
                'attention_mask': new_attention_mask[:, -max_len:],
            },
            non_tensors=dict(rollings.non_tensor_batch),
        )

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          tool_responses_ids: torch.Tensor) -> Dict:
        """Update right side state."""
        responses = self.tensor_fn.concatenate_with_padding([
            right_side['responses'],
            cur_responses,
            tool_responses_ids
        ], pad_to_left=False)

        old_lengths = self.tensor_fn.create_attention_mask(right_side['responses']).sum(dim=1)
        cur_lengths = self.tensor_fn.create_attention_mask(cur_responses).sum(dim=1)
        tool_lengths = self.tensor_fn.create_attention_mask(tool_responses_ids).sum(dim=1)
        response_mask = torch.zeros_like(responses)

        for i in range(responses.shape[0]):
            old_length = int(old_lengths[i].item())
            cur_length = int(cur_lengths[i].item())
            tool_length = int(tool_lengths[i].item())
            pieces = [right_side['response_mask'][i, :old_length]]
            if cur_length:
                pieces.append(torch.ones(cur_length, dtype=responses.dtype, device=responses.device))
            if tool_length:
                pieces.append(torch.zeros(tool_length, dtype=responses.dtype, device=responses.device))
            if pieces:
                merged_mask = torch.cat(pieces, dim=0)
                response_mask[i, :merged_mask.shape[0]] = merged_mask
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {
            'responses': responses[:, :max_len],
            'response_mask': response_mask[:, :max_len],
        }


    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles batch divisibility requirements.
            if num_gpus <= 1, return self.sequence_generator.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.sequence_generator.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.sequence_generator.generate_sequences(active_batch)
            
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_non_tensors = {}
        for k, v in active_batch.non_tensor_batch.items():
            padded_non_tensors[k] = np.concatenate([v, np.repeat(v[:1], padding_size, axis=0)], axis=0)

        padded_active_batch = DataProto.from_dict(tensors=padded_batch, non_tensors=padded_non_tensors)
        
        # Generate with padded batch
        padded_output = self.sequence_generator.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        padded_output.non_tensor_batch = {
            k: v[:-padding_size] for k, v in padded_output.non_tensor_batch.items()
        }
        return padded_output
    
    def run_llm_loop(self, gen_batch, envs: List[Any] = None,
                    initial_input_ids: torch.Tensor = None) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            'responses': initial_input_ids[:, []],
            'response_mask': initial_input_ids[:, []],
        }
        
        batch_size = gen_batch.batch['input_ids'].shape[0]
        
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns = torch.zeros(batch_size, dtype=torch.int32)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        meta_info = {}

        progress_log(
            f"rollout_start batch={batch_size} max_turns={self.config.max_turns} "
            f"max_prompt={self.config.max_prompt_length} max_response={self.config.max_response_length}"
        )

        # Main generation loop
        for step in range(self.config.max_turns):
            active_count = int(active_mask.sum().item())
            if not active_count:
                break
            progress_log(f"rollout_turn_start turn={step + 1}/{self.config.max_turns} active={active_count}")
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict(
                tensors={k: v[active_mask] for k, v in rollings.batch.items()},
                non_tensors={k: v[active_mask.cpu().numpy()] for k, v in rollings.non_tensor_batch.items()},
            )
            generate_start = time.perf_counter()
            with progress_heartbeat(
                f"rollout_turn_generate turn={step + 1}/{self.config.max_turns} active={active_count}"
            ):
                gen_output = self._generate_with_gpu_padding(rollings_active)
            generate_elapsed = time.perf_counter() - generate_start

            meta_info = gen_output.meta_info            
            responses_ids, responses_str, new_active_masks = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            active_mask[active_mask.clone()] = new_active_masks

            turns[active_mask] += 1
            tool_call_count = int(new_active_masks.sum().item())

            tool_start = time.perf_counter()
            if self.config.use_batch_tool_calls:
                # Use batch execution for tool calls
                tool_results = self._execute_tool_calls_batch(responses_str, envs, active_mask)
            else:
                # Use sequential execution for tool calls
                tool_results = self._execute_tool_calls(responses_str, envs, active_mask)
            if len(tool_results) == 2:
                tool_responses, env_continue_masks = tool_results
                raw_tool_responses = [""] * len(tool_responses)
            else:
                tool_responses, env_continue_masks, raw_tool_responses = tool_results
            tool_elapsed = time.perf_counter() - tool_start

            env_continue_masks = torch.tensor(env_continue_masks, dtype=torch.bool)
            active_mask = active_mask & env_continue_masks
            self._update_raw_prompts_for_next_turn(rollings, responses_str, raw_tool_responses, active_mask)

            continue_count = int(active_mask.sum().item())
            active_num_list.append(continue_count)
            progress_log(
                f"rollout_turn_done turn={step + 1}/{self.config.max_turns} "
                f"generated={active_count} tool_calls={tool_call_count} continue={continue_count} "
                f"generate_s={generate_elapsed:.2f} tool_s={tool_elapsed:.2f}"
            )
            tool_responses_ids = self._process_tool_responses(tool_responses)
            history_response_ids = self._responses_for_next_turn(responses_str)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                history_response_ids,
                tool_responses_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                tool_responses_ids
            )
        
        print("ACTIVE_TRAJ_NUM:", active_num_list, flush=True)
        progress_log(f"rollout_done active_path={active_num_list}")
        
        original_right_side['turns'] = turns
        termination_reason = np.array(
            [getattr(env, "termination_reason", None) for env in envs],
            dtype=object,
        )
        predict_calls_used = np.array(
            [getattr(env, "predict_calls_used", 0) for env in envs],
            dtype=object,
        )
        last_valid_tool_call = np.array(
            [getattr(env, "last_valid_tool_call", None) for env in envs],
            dtype=object,
        )
        last_valid_config = np.array(
            [getattr(env, "last_valid_config", None) for env in envs],
            dtype=object,
        )
        
        # Save trajectory and return final output
        return self._compose_final_output(
            original_left_side,
            original_right_side,
            meta_info,
            non_tensor_batch={
                "termination_reason": termination_reason,
                "predict_calls_used": predict_calls_used,
                "last_valid_tool_call": last_valid_tool_call,
                "last_valid_config": last_valid_config,
            },
        )


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            non_tensor_batch: Optional[Dict[str, Any]] = None) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']

        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        final_output['response_mask'] = right_side['response_mask']
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(
            tensors=final_output,
            non_tensors=non_tensor_batch or {},
        )
        final_output.meta_info.update(meta_info)

        return final_output
