"""
Code Generation Agent - Specialized in generating, reviewing, and explaining code
"""
import ast
import re
import subprocess
import tempfile
import os
import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent, AgentTask
from .utils import sanitize_input, generate_task_id

class ProgrammingLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    BASH = "bash"

@dataclass
class CodeSnippet:
    """Represents a code snippet"""
    id: str
    language: ProgrammingLanguage
    code: str
    description: str
    tags: List[str] = None
    is_tested: bool = False
    test_results: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class CodeReview:
    """Represents a code review"""
    id: str
    original_code: str
    language: ProgrammingLanguage
    issues: List[str]
    suggestions: List[str]
    rating: int  # 1-10 scale
    improved_code: Optional[str] = None

class CodeGenerationAgent(BaseAgent):
    """Code Generation Agent for programming tasks"""
    
    def __init__(self):
        super().__init__(
            name="code_generation",
            description="Specialized in generating, reviewing, and explaining code"
        )
        
        self.code_snippets: List[CodeSnippet] = []
        self.code_reviews: List[CodeReview] = []

        # Initialize specialized code generation pipeline
        self._initialize_code_pipeline()

        # Code templates for different languages
        self.templates = {
            ProgrammingLanguage.PYTHON: {
                "function": "def {name}({params}):\n    \"\"\"{docstring}\"\"\"\n    {body}\n    return {return_value}",
                "class": "class {name}:\n    \"\"\"{docstring}\"\"\"\n    \n    def __init__(self{params}):\n        {init_body}\n    \n    {methods}",
                "script": "#!/usr/bin/env python3\n\"\"\"\n{description}\n\"\"\"\n\n{imports}\n\n{main_code}\n\nif __name__ == '__main__':\n    {main_call}"
            },
            ProgrammingLanguage.JAVASCRIPT: {
                "function": "function {name}({params}) {{\n    // {description}\n    {body}\n    return {return_value};\n}}",
                "class": "class {name} {{\n    constructor({params}) {{\n        {constructor_body}\n    }}\n    \n    {methods}\n}}",
                "module": "// {description}\n\n{imports}\n\n{code}\n\nmodule.exports = {{{exports}}};"
            }
        }

    def _initialize_code_pipeline(self):
        """Initialize specialized code generation pipeline"""
        try:
            from transformers import T5ForConditionalGeneration, RobertaTokenizer

            # Check if we're using a CodeT5+ model
            model_name = self.config.get("model", "microsoft/DialoGPT-medium")

            if "codet5" in model_name.lower():
                self.logger.info(f"Initializing CodeT5+ pipeline for: {model_name}")

                # Initialize CodeT5+ model and tokenizer
                self.code_tokenizer = RobertaTokenizer.from_pretrained(model_name)
                self.code_model = T5ForConditionalGeneration.from_pretrained(model_name)

                # Set up for code generation
                self.code_pipeline_available = True
                self.logger.info("CodeT5+ pipeline initialized successfully")
            else:
                self.code_pipeline_available = False
                self.logger.info("Using default text generation pipeline")

        except Exception as e:
            self.logger.error(f"Failed to initialize code pipeline: {str(e)}")
            self.code_pipeline_available = False

    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response using specialized code model if available"""
        try:
            if hasattr(self, 'code_pipeline_available') and self.code_pipeline_available:
                return self._generate_with_codet5(prompt, max_length)
            else:
                # Fallback to parent method
                return super().generate_response(prompt, max_length)
        except Exception as e:
            self.logger.error(f"Code generation failed: {str(e)}")
            # Fallback to parent method
            return super().generate_response(prompt, max_length)

    def _generate_with_codet5(self, prompt: str, max_length: int = 512) -> str:
        """Generate code using CodeT5+ model with task-specific prompts"""
        try:
            # Analyze task type for better prompting
            task_type = self._analyze_task_type(prompt.lower())

            # Create task-specific prompt
            formatted_prompt = self._create_task_specific_prompt(prompt, task_type)

            # Tokenize input
            inputs = self.code_tokenizer.encode(
                formatted_prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True
            )

            # Adjust generation parameters based on task type
            generation_params = self._get_generation_params(task_type, max_length)

            # Generate code
            with torch.no_grad():
                outputs = self.code_model.generate(
                    inputs,
                    **generation_params,
                    pad_token_id=self.code_tokenizer.pad_token_id,
                    eos_token_id=self.code_tokenizer.eos_token_id
                )

            # Decode generated code
            generated_code = self.code_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean up the generated code
            generated_code = self._clean_generated_code(generated_code, formatted_prompt)

            # Post-process based on task type
            generated_code = self._post_process_code(generated_code, task_type)

            return generated_code

        except Exception as e:
            self.logger.error(f"CodeT5+ generation failed: {str(e)}")
            raise

    def _create_task_specific_prompt(self, prompt: str, task_type: str) -> str:
        """Create task-specific prompts for better code generation"""
        base_prompt = prompt.strip()

        if task_type == "api_development":
            return f"Create a Python REST API: {base_prompt}. Include proper error handling, request validation, and response formatting."

        elif task_type == "data_processing":
            return f"Write Python code for data processing: {base_prompt}. Include error handling, data validation, and proper documentation."

        elif task_type == "algorithm":
            return f"Implement a Python algorithm: {base_prompt}. Include time complexity analysis, edge case handling, and example usage."

        elif task_type == "object_oriented":
            return f"Create Python classes: {base_prompt}. Include proper encapsulation, inheritance if needed, and comprehensive methods."

        elif task_type == "file_io":
            return f"Write Python file I/O code: {base_prompt}. Include proper exception handling, file closing, and encoding support."

        elif task_type == "database":
            return f"Create Python database code: {base_prompt}. Include connection management, error handling, and proper SQL practices."

        elif task_type == "testing":
            return f"Write Python unit tests: {base_prompt}. Include test cases, assertions, mocking, and proper test structure."

        elif task_type == "machine_learning":
            return f"Create Python ML code: {base_prompt}. Include data preprocessing, model training, evaluation, and proper ML practices."

        elif task_type == "web_development":
            return f"Write Python web application code: {base_prompt}. Include routing, templates, error handling, and security considerations."

        elif task_type == "concurrency":
            return f"Create Python concurrent code: {base_prompt}. Include proper thread/async handling, synchronization, and error management."

        elif task_type == "networking":
            return f"Write Python networking code: {base_prompt}. Include socket handling, error management, and protocol implementation."

        elif task_type == "gui":
            return f"Create Python GUI application: {base_prompt}. Include event handling, layout management, and user interaction."

        elif task_type == "game_development":
            return f"Write Python game code: {base_prompt}. Include game loop, event handling, and proper game structure."

        elif task_type == "mathematical":
            return f"Implement Python mathematical code: {base_prompt}. Include numerical accuracy, edge cases, and mathematical documentation."

        else:
            return f"Generate Python code: {base_prompt}. Include proper error handling, documentation, and example usage."

    def _get_generation_params(self, task_type: str, max_length: int) -> dict:
        """Get generation parameters based on task type"""
        base_params = {
            "max_length": max_length,
            "num_beams": 5,
            "temperature": 0.2,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }

        # Adjust parameters based on task complexity
        if task_type in ["algorithm", "machine_learning", "database"]:
            # More complex tasks need more diverse generation
            base_params.update({
                "temperature": 0.3,
                "top_p": 0.95,
                "num_beams": 3
            })

        elif task_type in ["api_development", "web_development"]:
            # Structured code needs more deterministic generation
            base_params.update({
                "temperature": 0.1,
                "top_p": 0.8,
                "num_beams": 7
            })

        elif task_type in ["testing", "file_io"]:
            # Standard patterns benefit from balanced generation
            base_params.update({
                "temperature": 0.15,
                "top_p": 0.85,
                "num_beams": 5
            })

        return base_params

    def _clean_generated_code(self, generated_code: str, prompt: str) -> str:
        """Clean up generated code"""
        # Remove the prompt from generated code
        if prompt in generated_code:
            generated_code = generated_code.replace(prompt, "").strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "Generate Python code:",
            "Create a Python",
            "Write Python",
            "Implement a Python",
            "Here's the code:",
            "Here is the code:",
            "```python",
            "```",
            "python"
        ]

        for prefix in prefixes_to_remove:
            if generated_code.startswith(prefix):
                generated_code = generated_code[len(prefix):].strip()

        # Remove trailing artifacts
        suffixes_to_remove = ["```", "```python", "```py"]
        for suffix in suffixes_to_remove:
            if generated_code.endswith(suffix):
                generated_code = generated_code[:-len(suffix)].strip()

        return generated_code

    def _post_process_code(self, code: str, task_type: str) -> str:
        """Post-process generated code based on task type"""
        if not code.strip():
            return code

        # Add task-specific improvements
        if task_type == "api_development" and "if __name__" not in code:
            code += '\n\nif __name__ == "__main__":\n    # Run the application\n    print("API server ready to start")'

        elif task_type == "testing" and "unittest.main()" not in code:
            code += '\n\nif __name__ == "__main__":\n    unittest.main()'

        elif task_type == "algorithm" and "# Example usage" not in code and "if __name__" not in code:
            code += '\n\n# Example usage\nif __name__ == "__main__":\n    # Test the algorithm\n    pass'

        elif task_type == "data_processing" and "if __name__" not in code:
            code += '\n\nif __name__ == "__main__":\n    # Example data processing\n    pass'

        # Ensure proper indentation
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            # Fix common indentation issues
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                if any(line.startswith(keyword) for keyword in ['def ', 'class ', 'import ', 'from ']):
                    cleaned_lines.append(line)
                elif cleaned_lines and cleaned_lines[-1].endswith(':'):
                    cleaned_lines.append('    ' + line)
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    async def _process_task_impl(self, task: AgentTask) -> str:
        """Process code generation tasks"""
        task_type = task.task_type.lower()
        content = sanitize_input(task.content)
        
        if task_type == "generate_code":
            return await self._generate_code(content, task.metadata)
        elif task_type == "review_code":
            return await self._review_code(content, task.metadata)
        elif task_type == "explain_code":
            return await self._explain_code(content, task.metadata)
        elif task_type == "debug_code":
            return await self._debug_code(content, task.metadata)
        elif task_type == "optimize_code":
            return await self._optimize_code(content, task.metadata)
        elif task_type == "convert_code":
            return await self._convert_code(content, task.metadata)
        elif task_type == "generate_tests":
            return await self._generate_tests(content, task.metadata)
        elif task_type == "code_documentation":
            return await self._generate_documentation(content, task.metadata)
        elif task_type == "health_check":
            return "Code Generation Agent is ready to help with all your programming needs!"
        else:
            return await self._generate_code(content, task.metadata)
    
    async def _generate_code(self, description: str, metadata: Dict[str, Any]) -> str:
        """Generate code based on description"""
        try:
            language = self._get_language_from_metadata(metadata)
            code_type = metadata.get("code_type", "function")  # function, class, script

            # Check for simple/common requests and provide immediate responses
            simple_code = self._get_simple_code_template(description, language)
            if simple_code:
                cleaned_code = simple_code
            else:
                # Create a detailed prompt for code generation
                prompt = self._create_code_generation_prompt(description, language, code_type, metadata)

                # Generate code using LLM
                generated_code = self.generate_response(prompt, max_length=1024)

                # Clean and format the generated code
                cleaned_code = self._clean_generated_code(generated_code, language)

                # If cleaned code is empty or too short, use fallback
                if not cleaned_code or len(cleaned_code.strip()) < 10:
                    cleaned_code = self._get_fallback_code(description, language)

            # Validate the generated code
            validation_result = self._validate_code(cleaned_code, language)

            # Create code snippet record
            snippet = CodeSnippet(
                id=generate_task_id(),
                language=language,
                code=cleaned_code,
                description=description,
                tags=metadata.get("tags", [])
            )
            self.code_snippets.append(snippet)

            result = f"Generated {language.value} code:\n\n```{language.value}\n{cleaned_code}\n```"

            if validation_result:
                result += f"\n\n✅ Code validation: {validation_result}"

            # Add usage example if requested
            if metadata.get("include_example", False):
                example = self._generate_usage_example(cleaned_code, language)
                if example:
                    result += f"\n\nUsage example:\n```{language.value}\n{example}\n```"

            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate code: {str(e)}")
            return f"Sorry, I couldn't generate the code. Error: {str(e)}"
    
    async def _review_code(self, code: str, metadata: Dict[str, Any]) -> str:
        """Review code and provide feedback"""
        try:
            language = self._get_language_from_metadata(metadata)
            
            # Analyze the code
            issues = self._analyze_code_issues(code, language)
            suggestions = self._generate_code_suggestions(code, language)
            rating = self._rate_code_quality(code, language)
            
            # Generate improved version if requested
            improved_code = None
            if metadata.get("generate_improved", False):
                improved_code = await self._improve_code(code, issues, suggestions, language)
            
            # Create review record
            review = CodeReview(
                id=generate_task_id(),
                original_code=code,
                language=language,
                issues=issues,
                suggestions=suggestions,
                rating=rating,
                improved_code=improved_code
            )
            self.code_reviews.append(review)
            
            # Format review response
            result = f"Code Review Results:\n\n"
            result += f"**Quality Rating:** {rating}/10\n\n"
            
            if issues:
                result += "**Issues Found:**\n"
                for i, issue in enumerate(issues, 1):
                    result += f"{i}. {issue}\n"
                result += "\n"
            
            if suggestions:
                result += "**Suggestions for Improvement:**\n"
                for i, suggestion in enumerate(suggestions, 1):
                    result += f"{i}. {suggestion}\n"
                result += "\n"
            
            if improved_code:
                result += f"**Improved Code:**\n```{language.value}\n{improved_code}\n```"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to review code: {str(e)}")
            return f"Sorry, I couldn't review the code. Error: {str(e)}"
    
    async def _explain_code(self, code: str, metadata: Dict[str, Any]) -> str:
        """Explain how code works"""
        try:
            language = self._get_language_from_metadata(metadata)
            detail_level = metadata.get("detail_level", "medium")  # basic, medium, detailed
            
            # Create explanation prompt
            prompt = f"""
            Please explain this {language.value} code in {detail_level} detail:
            
            ```{language.value}
            {code}
            ```
            
            Explain:
            1. What the code does overall
            2. How each part works
            3. Key concepts used
            4. Any potential issues or improvements
            """
            
            explanation = self.generate_response(prompt, max_length=1024)
            
            # Add code structure analysis
            structure_analysis = self._analyze_code_structure(code, language)
            
            result = f"Code Explanation:\n\n{explanation}"
            
            if structure_analysis:
                result += f"\n\n**Code Structure Analysis:**\n{structure_analysis}"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to explain code: {str(e)}")
            return f"Sorry, I couldn't explain the code. Error: {str(e)}"
    
    async def _debug_code(self, code: str, metadata: Dict[str, Any]) -> str:
        """Debug code and find issues"""
        try:
            language = self._get_language_from_metadata(metadata)
            error_message = metadata.get("error_message", "")
            
            # Analyze for common bugs
            bugs = self._find_common_bugs(code, language)
            
            # If there's an error message, analyze it
            error_analysis = ""
            if error_message:
                error_analysis = self._analyze_error_message(error_message, code, language)
            
            # Generate debugging suggestions
            debug_suggestions = self._generate_debug_suggestions(code, language, error_message)
            
            # Try to provide a fixed version
            fixed_code = self._attempt_code_fix(code, language, bugs, error_message)
            
            result = "Debug Analysis:\n\n"
            
            if error_analysis:
                result += f"**Error Analysis:**\n{error_analysis}\n\n"
            
            if bugs:
                result += "**Potential Issues Found:**\n"
                for i, bug in enumerate(bugs, 1):
                    result += f"{i}. {bug}\n"
                result += "\n"
            
            if debug_suggestions:
                result += "**Debugging Suggestions:**\n"
                for i, suggestion in enumerate(debug_suggestions, 1):
                    result += f"{i}. {suggestion}\n"
                result += "\n"
            
            if fixed_code and fixed_code != code:
                result += f"**Suggested Fix:**\n```{language.value}\n{fixed_code}\n```"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to debug code: {str(e)}")
            return f"Sorry, I couldn't debug the code. Error: {str(e)}"
    
    async def _optimize_code(self, code: str, metadata: Dict[str, Any]) -> str:
        """Optimize code for performance and readability"""
        try:
            language = self._get_language_from_metadata(metadata)
            optimization_type = metadata.get("optimization_type", "both")  # performance, readability, both
            
            optimizations = []
            optimized_code = code
            
            if optimization_type in ["performance", "both"]:
                perf_optimizations = self._find_performance_optimizations(code, language)
                optimizations.extend(perf_optimizations)
                optimized_code = self._apply_performance_optimizations(optimized_code, perf_optimizations, language)
            
            if optimization_type in ["readability", "both"]:
                readability_improvements = self._find_readability_improvements(code, language)
                optimizations.extend(readability_improvements)
                optimized_code = self._apply_readability_improvements(optimized_code, readability_improvements, language)
            
            result = "Code Optimization Results:\n\n"
            
            if optimizations:
                result += "**Optimizations Applied:**\n"
                for i, opt in enumerate(optimizations, 1):
                    result += f"{i}. {opt}\n"
                result += "\n"
            
            if optimized_code != code:
                result += f"**Optimized Code:**\n```{language.value}\n{optimized_code}\n```"
            else:
                result += "The code is already well-optimized!"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to optimize code: {str(e)}")
            return f"Sorry, I couldn't optimize the code. Error: {str(e)}"
    
    async def _convert_code(self, code: str, metadata: Dict[str, Any]) -> str:
        """Convert code from one language to another"""
        try:
            source_language = self._get_language_from_metadata(metadata, "source_language")
            target_language = self._get_language_from_metadata(metadata, "target_language")
            
            if source_language == target_language:
                return "Source and target languages are the same!"
            
            # Create conversion prompt
            prompt = f"""
            Convert this {source_language.value} code to {target_language.value}:
            
            ```{source_language.value}
            {code}
            ```
            
            Maintain the same functionality and logic. Provide clean, idiomatic {target_language.value} code.
            """
            
            converted_code = self.generate_response(prompt, max_length=1024)
            cleaned_converted = self._clean_generated_code(converted_code, target_language)
            
            result = f"Code Conversion ({source_language.value} → {target_language.value}):\n\n"
            result += f"```{target_language.value}\n{cleaned_converted}\n```\n\n"
            
            # Add conversion notes
            conversion_notes = self._generate_conversion_notes(source_language, target_language)
            if conversion_notes:
                result += f"**Conversion Notes:**\n{conversion_notes}"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to convert code: {str(e)}")
            return f"Sorry, I couldn't convert the code. Error: {str(e)}"
    
    async def _generate_tests(self, code: str, metadata: Dict[str, Any]) -> str:
        """Generate unit tests for the given code"""
        try:
            language = self._get_language_from_metadata(metadata)
            test_framework = metadata.get("test_framework", self._get_default_test_framework(language))
            
            # Analyze code to understand what to test
            test_cases = self._identify_test_cases(code, language)
            
            # Generate test code
            test_code = self._generate_test_code(code, test_cases, language, test_framework)
            
            result = f"Generated Unit Tests ({test_framework}):\n\n"
            result += f"```{language.value}\n{test_code}\n```\n\n"
            
            # Add test execution instructions
            instructions = self._get_test_execution_instructions(language, test_framework)
            if instructions:
                result += f"**How to run tests:**\n{instructions}"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate tests: {str(e)}")
            return f"Sorry, I couldn't generate tests. Error: {str(e)}"
    
    async def _generate_documentation(self, code: str, metadata: Dict[str, Any]) -> str:
        """Generate documentation for the code"""
        try:
            language = self._get_language_from_metadata(metadata)
            doc_style = metadata.get("doc_style", "standard")  # standard, detailed, api
            
            # Analyze code structure
            functions = self._extract_functions(code, language)
            classes = self._extract_classes(code, language)
            
            # Generate documentation
            documentation = self._create_documentation(code, functions, classes, language, doc_style)
            
            result = f"Generated Documentation:\n\n{documentation}"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate documentation: {str(e)}")
            return f"Sorry, I couldn't generate documentation. Error: {str(e)}"
    
    # Helper methods
    def _get_language_from_metadata(self, metadata: Dict[str, Any], key: str = "language") -> ProgrammingLanguage:
        """Extract programming language from metadata"""
        lang_str = metadata.get(key, "python").lower()
        try:
            return ProgrammingLanguage(lang_str)
        except ValueError:
            return ProgrammingLanguage.PYTHON  # Default fallback
    
    def _create_code_generation_prompt(self, description: str, language: ProgrammingLanguage, 
                                     code_type: str, metadata: Dict[str, Any]) -> str:
        """Create a detailed prompt for code generation"""
        prompt = f"""
        Generate {language.value} code for the following requirement:
        
        Description: {description}
        Code Type: {code_type}
        
        Requirements:
        - Write clean, readable, and well-commented code
        - Follow {language.value} best practices and conventions
        - Include proper error handling where appropriate
        - Make the code modular and reusable
        """
        
        if metadata.get("include_docstrings", True):
            prompt += "\n- Include comprehensive docstrings/comments"
        
        if metadata.get("include_type_hints", False) and language == ProgrammingLanguage.PYTHON:
            prompt += "\n- Include type hints"
        
        return prompt
    
    def _clean_generated_code(self, generated_code: str, language: ProgrammingLanguage) -> str:
        """Clean and format generated code"""
        # Remove markdown code blocks if present
        code = re.sub(r'```\w*\n?', '', generated_code)
        code = re.sub(r'```', '', code)
        
        # Remove common prefixes from LLM responses
        code = re.sub(r'^(Here\'s|Here is|The code is).*?:\s*', '', code, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up extra whitespace
        lines = [line.rstrip() for line in code.split('\n')]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        return '\n'.join(lines)

    def _get_simple_code_template(self, description: str, language: ProgrammingLanguage) -> str:
        """Get simple code templates for specific tasks"""
        desc_lower = description.lower()

        # Analyze the task type for better template selection
        task_type = self._analyze_task_type(desc_lower)

        # Python templates based on task type
        if language == ProgrammingLanguage.PYTHON:
            # Task-specific templates
            if task_type == "api_development":
                return self._get_api_template(desc_lower)
            elif task_type == "data_processing":
                return self._get_data_processing_template(desc_lower)
            elif task_type == "file_io":
                return self._get_file_io_template(desc_lower)
            elif task_type == "algorithm":
                return self._get_algorithm_template(desc_lower)
            elif task_type == "object_oriented":
                return self._get_oop_template(desc_lower)
            elif task_type == "database":
                return self._get_database_template(desc_lower)
            elif task_type == "testing":
                return self._get_testing_template(desc_lower)
            elif task_type == "machine_learning":
                return self._get_ml_template(desc_lower)
            elif task_type == "web_development":
                return self._get_web_template(desc_lower)
            elif task_type == "concurrency":
                return self._get_concurrency_template(desc_lower)

            # Basic templates for simple requests
            elif "hello world" in desc_lower or "hello" in desc_lower:
                return '''def hello_world():
    """Print a hello world message."""
    print("Hello, World!")
    return "Hello, World!"

# Call the function
if __name__ == "__main__":
    hello_world()'''

            elif "factorial" in desc_lower:
                return '''def factorial(n):
    """Calculate the factorial of a number."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1

    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# Example usage
if __name__ == "__main__":
    print(f"5! = {factorial(5)}")'''

            elif "fibonacci" in desc_lower:
                return '''def fibonacci(n):
    """Generate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
if __name__ == "__main__":
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")'''

            elif "rest api" in desc_lower or "api" in desc_lower:
                return '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Create FastAPI app
app = FastAPI(title="My REST API", version="1.0.0")

# Data models
class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    price: float

class ItemResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float

# In-memory storage (use database in production)
items_db = []
next_id = 1

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to My REST API"}

@app.get("/items", response_model=List[ItemResponse])
async def get_items():
    """Get all items"""
    return items_db

@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    """Get a specific item by ID"""
    for item in items_db:
        if item["id"] == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")

@app.post("/items", response_model=ItemResponse)
async def create_item(item: Item):
    """Create a new item"""
    global next_id
    new_item = {
        "id": next_id,
        "name": item.name,
        "description": item.description,
        "price": item.price
    }
    items_db.append(new_item)
    next_id += 1
    return new_item

@app.put("/items/{item_id}", response_model=ItemResponse)
async def update_item(item_id: int, item: Item):
    """Update an existing item"""
    for i, existing_item in enumerate(items_db):
        if existing_item["id"] == item_id:
            updated_item = {
                "id": item_id,
                "name": item.name,
                "description": item.description,
                "price": item.price
            }
            items_db[i] = updated_item
            return updated_item
    raise HTTPException(status_code=404, detail="Item not found")

@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    """Delete an item"""
    for i, item in enumerate(items_db):
        if item["id"] == item_id:
            del items_db[i]
            return {"message": "Item deleted successfully"}
    raise HTTPException(status_code=404, detail="Item not found")

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)'''

            elif "sum" in desc_lower and ("input" in desc_lower or "two" in desc_lower):
                return '''def add_two_numbers():
    """Function to take two inputs and print their sum."""
    try:
        # Get input from user
        num1 = float(input("Enter the first number: "))
        num2 = float(input("Enter the second number: "))

        # Calculate sum
        result = num1 + num2

        # Print the result
        print(f"The sum of {num1} and {num2} is: {result}")

        return result

    except ValueError:
        print("Error: Please enter valid numbers!")
        return None

# Alternative version with parameters
def add_numbers(a, b):
    """Add two numbers and return the result."""
    result = a + b
    print(f"The sum of {a} and {b} is: {result}")
    return result

# Example usage
if __name__ == "__main__":
    # Version 1: Interactive input
    print("Version 1 - Interactive input:")
    add_two_numbers()

    print("\\nVersion 2 - Function parameters:")
    add_numbers(5, 3)
    add_numbers(10.5, 7.2)'''

            elif "calculator" in desc_lower:
                return '''def calculator():
    """Simple calculator function."""
    print("Simple Calculator")
    print("Operations: +, -, *, /")

    try:
        num1 = float(input("Enter first number: "))
        operation = input("Enter operation (+, -, *, /): ")
        num2 = float(input("Enter second number: "))

        if operation == '+':
            result = num1 + num2
        elif operation == '-':
            result = num1 - num2
        elif operation == '*':
            result = num1 * num2
        elif operation == '/':
            if num2 != 0:
                result = num1 / num2
            else:
                print("Error: Division by zero!")
                return None
        else:
            print("Error: Invalid operation!")
            return None

        print(f"{num1} {operation} {num2} = {result}")
        return result

    except ValueError:
        print("Error: Please enter valid numbers!")
        return None

# Example usage
if __name__ == "__main__":
    calculator()'''

            elif "function" in desc_lower and ("add" in desc_lower or "plus" in desc_lower):
                return '''def add_numbers(a, b):
    """Add two numbers and return the result."""
    result = a + b
    print(f"The sum of {a} and {b} is: {result}")
    return result

# Example usage
if __name__ == "__main__":
    # Test the function
    result1 = add_numbers(5, 3)
    result2 = add_numbers(10.5, 7.2)
    result3 = add_numbers(-4, 9)

    print(f"Results: {result1}, {result2}, {result3}")'''

            elif "list" in desc_lower and ("sort" in desc_lower or "order" in desc_lower):
                return '''def sort_list(numbers):
    """Sort a list of numbers and return the sorted list."""
    sorted_numbers = sorted(numbers)
    print(f"Original list: {numbers}")
    print(f"Sorted list: {sorted_numbers}")
    return sorted_numbers

# Example usage
if __name__ == "__main__":
    # Test with different lists
    test_list1 = [64, 34, 25, 12, 22, 11, 90]
    test_list2 = [3.14, 2.71, 1.41, 1.73]

    sorted1 = sort_list(test_list1)
    sorted2 = sort_list(test_list2)'''

            elif "loop" in desc_lower or "iterate" in desc_lower:
                return '''def print_numbers(start, end):
    """Print numbers from start to end using a loop."""
    print(f"Numbers from {start} to {end}:")

    for i in range(start, end + 1):
        print(i, end=" ")
    print()  # New line

    return list(range(start, end + 1))

def print_list_items(items):
    """Print each item in a list."""
    print("List items:")
    for index, item in enumerate(items):
        print(f"{index + 1}. {item}")

# Example usage
if __name__ == "__main__":
    # Print numbers in range
    numbers = print_numbers(1, 10)

    # Print list items
    fruits = ["apple", "banana", "orange", "grape"]
    print_list_items(fruits)'''

        # JavaScript templates
        elif language == ProgrammingLanguage.JAVASCRIPT:
            if "hello world" in desc_lower or "hello" in desc_lower:
                return '''function helloWorld() {
    console.log("Hello, World!");
    return "Hello, World!";
}

// Call the function
helloWorld();'''

            elif "rest api" in desc_lower or "api" in desc_lower:
                return '''const express = require('express');
const cors = require('cors');

// Create Express app
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// In-memory storage (use database in production)
let items = [];
let nextId = 1;

// Routes
app.get('/', (req, res) => {
    res.json({ message: 'Welcome to Express REST API' });
});

app.get('/items', (req, res) => {
    res.json(items);
});

app.get('/items/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const item = items.find(item => item.id === id);

    if (!item) {
        return res.status(404).json({ error: 'Item not found' });
    }

    res.json(item);
});

app.post('/items', (req, res) => {
    const { name, description, price } = req.body;

    if (!name) {
        return res.status(400).json({ error: 'Name is required' });
    }

    const newItem = {
        id: nextId++,
        name,
        description: description || '',
        price: price || 0
    };

    items.push(newItem);
    res.status(201).json(newItem);
});

app.put('/items/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const itemIndex = items.findIndex(item => item.id === id);

    if (itemIndex === -1) {
        return res.status(404).json({ error: 'Item not found' });
    }

    const { name, description, price } = req.body;
    items[itemIndex] = {
        ...items[itemIndex],
        name: name || items[itemIndex].name,
        description: description || items[itemIndex].description,
        price: price || items[itemIndex].price
    };

    res.json(items[itemIndex]);
});

app.delete('/items/:id', (req, res) => {
    const id = parseInt(req.params.id);
    const itemIndex = items.findIndex(item => item.id === id);

    if (itemIndex === -1) {
        return res.status(404).json({ error: 'Item not found' });
    }

    items.splice(itemIndex, 1);
    res.json({ message: 'Item deleted successfully' });
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});'''

            elif "flask" in desc_lower and "api" in desc_lower:
                return '''from flask import Flask, request, jsonify
from flask_cors import CORS

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# In-memory storage (use database in production)
items = []
next_id = 1

@app.route('/')
def home():
    """Root endpoint"""
    return jsonify({"message": "Welcome to Flask REST API"})

@app.route('/items', methods=['GET'])
def get_items():
    """Get all items"""
    return jsonify(items)

@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    """Get a specific item by ID"""
    item = next((item for item in items if item['id'] == item_id), None)
    if item:
        return jsonify(item)
    return jsonify({"error": "Item not found"}), 404

@app.route('/items', methods=['POST'])
def create_item():
    """Create a new item"""
    global next_id
    data = request.get_json()

    if not data or 'name' not in data:
        return jsonify({"error": "Name is required"}), 400

    new_item = {
        "id": next_id,
        "name": data['name'],
        "description": data.get('description', ''),
        "price": data.get('price', 0.0)
    }
    items.append(new_item)
    next_id += 1
    return jsonify(new_item), 201

@app.route('/items/<int:item_id>', methods=['PUT'])
def update_item(item_id):
    """Update an existing item"""
    data = request.get_json()
    item = next((item for item in items if item['id'] == item_id), None)

    if not item:
        return jsonify({"error": "Item not found"}), 404

    item['name'] = data.get('name', item['name'])
    item['description'] = data.get('description', item['description'])
    item['price'] = data.get('price', item['price'])

    return jsonify(item)

@app.route('/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    """Delete an item"""
    global items
    items = [item for item in items if item['id'] != item_id]
    return jsonify({"message": "Item deleted successfully"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)'''

        # JavaScript templates
        elif language == ProgrammingLanguage.JAVASCRIPT:
            if "hello world" in desc_lower or "hello" in desc_lower:
                return '''function helloWorld() {
    console.log("Hello, World!");
    return "Hello, World!";
}

// Call the function
helloWorld();'''

        return None

    def _analyze_task_type(self, description: str) -> str:
        """Analyze the task type for better template selection"""
        # Task type detection based on keywords
        if any(kw in description for kw in ["api", "rest", "endpoint", "server", "http"]):
            return "api_development"
        elif any(kw in description for kw in ["data", "process", "transform", "clean", "parse"]):
            return "data_processing"
        elif any(kw in description for kw in ["class", "object", "inherit", "method"]):
            return "object_oriented"
        elif any(kw in description for kw in ["algorithm", "sort", "search", "tree", "graph"]):
            return "algorithm"
        elif any(kw in description for kw in ["file", "read", "write", "open", "save", "load"]):
            return "file_io"
        elif any(kw in description for kw in ["web", "html", "css", "javascript", "dom"]):
            return "web_development"
        elif any(kw in description for kw in ["database", "sql", "query", "table", "orm"]):
            return "database"
        elif any(kw in description for kw in ["test", "unit test", "mock", "assert"]):
            return "testing"
        elif any(kw in description for kw in ["gui", "interface", "button", "window", "ui"]):
            return "gui"
        elif any(kw in description for kw in ["machine learning", "ml", "model", "train", "predict"]):
            return "machine_learning"
        elif any(kw in description for kw in ["game", "pygame", "sprite", "collision"]):
            return "game_development"
        elif any(kw in description for kw in ["math", "calculate", "compute", "formula"]):
            return "mathematical"
        elif any(kw in description for kw in ["network", "socket", "tcp", "udp", "client", "server"]):
            return "networking"
        elif any(kw in description for kw in ["thread", "async", "concurrent", "parallel"]):
            return "concurrency"
        else:
            return "general"

    def _get_api_template(self, description: str) -> str:
        """Get API development templates"""
        if "fastapi" in description or "fast api" in description:
            return '''from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="My API", version="1.0.0")

# Data models
class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None

class ItemResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

# In-memory storage
items_db = []
next_id = 1

@app.get("/")
async def root():
    return {"message": "Welcome to My API"}

@app.get("/items", response_model=List[ItemResponse])
async def get_items():
    return items_db

@app.post("/items", response_model=ItemResponse)
async def create_item(item: Item):
    global next_id
    new_item = ItemResponse(
        id=next_id,
        name=item.name,
        description=item.description
    )
    items_db.append(new_item.dict())
    next_id += 1
    return new_item

@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    for item in items_db:
        if item["id"] == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)'''

        elif "flask" in description:
            return '''from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# In-memory storage
items = []
next_id = 1

@app.route('/')
def home():
    return jsonify({"message": "Welcome to Flask API"})

@app.route('/items', methods=['GET'])
def get_items():
    return jsonify(items)

@app.route('/items', methods=['POST'])
def create_item():
    global next_id
    data = request.get_json()

    if not data or 'name' not in data:
        return jsonify({"error": "Name is required"}), 400

    new_item = {
        "id": next_id,
        "name": data['name'],
        "description": data.get('description', '')
    }
    items.append(new_item)
    next_id += 1
    return jsonify(new_item), 201

@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    item = next((item for item in items if item['id'] == item_id), None)
    if item:
        return jsonify(item)
    return jsonify({"error": "Item not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)'''

        else:
            return '''import http.server
import socketserver
import json
from urllib.parse import urlparse, parse_qs

class APIHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"message": "Welcome to Simple API"}
            self.wfile.write(json.dumps(response).encode())

        elif parsed_path.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "healthy"}
            self.wfile.write(json.dumps(response).encode())

        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"error": "Not found"}
            self.wfile.write(json.dumps(response).encode())

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode('utf-8'))
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"message": "Data received", "data": data}
            self.wfile.write(json.dumps(response).encode())
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"error": "Invalid JSON"}
            self.wfile.write(json.dumps(response).encode())

if __name__ == "__main__":
    PORT = 8000
    with socketserver.TCPServer(("", PORT), APIHandler) as httpd:
        print(f"Server running at http://localhost:{PORT}")
        httpd.serve_forever()'''

    def _get_data_processing_template(self, description: str) -> str:
        """Get data processing templates"""
        if "pandas" in description or "dataframe" in description:
            return '''import pandas as pd
import numpy as np

def process_data(file_path):
    """Process data from CSV file using pandas."""
    try:
        # Read data
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

        # Basic data info
        print("\\nData Info:")
        print(df.info())

        # Handle missing values
        df_cleaned = df.dropna()  # or df.fillna(method='forward')

        # Basic statistics
        print("\\nBasic Statistics:")
        print(df_cleaned.describe())

        # Data transformation example
        if 'date' in df_cleaned.columns:
            df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])

        # Group by example (adjust column names as needed)
        if len(df_cleaned.columns) > 1:
            first_col = df_cleaned.columns[0]
            if df_cleaned[first_col].dtype in ['object', 'category']:
                grouped = df_cleaned.groupby(first_col).size()
                print(f"\\nGrouped by {first_col}:")
                print(grouped)

        # Save processed data
        output_path = file_path.replace('.csv', '_processed.csv')
        df_cleaned.to_csv(output_path, index=False)
        print(f"\\nProcessed data saved to: {output_path}")

        return df_cleaned

    except Exception as e:
        print(f"Error processing data: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Replace with your CSV file path
    data = process_data("data.csv")
    if data is not None:
        print("\\nFirst 5 rows:")
        print(data.head())'''

        else:
            return '''def process_data(data):
    """Generic data processing function."""
    processed_data = []

    for item in data:
        # Clean and transform data
        if isinstance(item, str):
            # String processing
            cleaned_item = item.strip().lower()
            processed_data.append(cleaned_item)
        elif isinstance(item, (int, float)):
            # Numeric processing
            processed_data.append(item * 2)  # Example transformation
        elif isinstance(item, dict):
            # Dictionary processing
            cleaned_dict = {k.strip(): v for k, v in item.items() if v is not None}
            processed_data.append(cleaned_dict)
        else:
            processed_data.append(item)

    return processed_data

def filter_data(data, condition_func):
    """Filter data based on condition function."""
    return [item for item in data if condition_func(item)]

def aggregate_data(data, key_func, value_func):
    """Aggregate data by key and value functions."""
    result = {}
    for item in data:
        key = key_func(item)
        value = value_func(item)

        if key in result:
            result[key].append(value)
        else:
            result[key] = [value]

    return result

# Example usage
if __name__ == "__main__":
    sample_data = [
        {"name": "Alice", "age": 30, "city": "New York"},
        {"name": "Bob", "age": 25, "city": "Los Angeles"},
        {"name": "Charlie", "age": 35, "city": "New York"}
    ]

    # Process data
    processed = process_data(sample_data)
    print("Processed data:", processed)

    # Filter adults over 25
    adults = filter_data(sample_data, lambda x: x["age"] > 25)
    print("Adults over 25:", adults)

    # Aggregate by city
    by_city = aggregate_data(sample_data, lambda x: x["city"], lambda x: x["name"])
    print("People by city:", by_city)'''

    def _get_file_io_template(self, description: str) -> str:
        """Get file I/O templates"""
        if "json" in description:
            return '''import json
import os

def read_json_file(file_path):
    """Read data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"Successfully read {len(data)} items from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def write_json_file(data, file_path):
    """Write data to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print(f"Successfully wrote data to {file_path}")
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False

def update_json_file(file_path, new_data):
    """Update existing JSON file with new data."""
    existing_data = read_json_file(file_path)
    if existing_data is not None:
        if isinstance(existing_data, list):
            existing_data.extend(new_data if isinstance(new_data, list) else [new_data])
        elif isinstance(existing_data, dict):
            existing_data.update(new_data)

        return write_json_file(existing_data, file_path)
    return False

# Example usage
if __name__ == "__main__":
    sample_data = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25}
    ]

    # Write data
    write_json_file(sample_data, "data.json")

    # Read data
    loaded_data = read_json_file("data.json")
    print("Loaded data:", loaded_data)

    # Update data
    new_person = {"name": "Charlie", "age": 35}
    update_json_file("data.json", new_person)'''

        elif "csv" in description:
            return '''import csv
import os

def read_csv_file(file_path, delimiter=','):
    """Read data from CSV file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file, delimiter=delimiter)
            for row in csv_reader:
                data.append(row)
        print(f"Successfully read {len(data)} rows from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def write_csv_file(data, file_path, fieldnames=None):
    """Write data to CSV file."""
    if not data:
        print("No data to write")
        return False

    try:
        if fieldnames is None:
            fieldnames = data[0].keys() if isinstance(data[0], dict) else range(len(data[0]))

        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        print(f"Successfully wrote {len(data)} rows to {file_path}")
        return True
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return False

def filter_csv_data(data, filter_func):
    """Filter CSV data based on condition."""
    return [row for row in data if filter_func(row)]

# Example usage
if __name__ == "__main__":
    sample_data = [
        {"name": "Alice", "age": "30", "city": "New York"},
        {"name": "Bob", "age": "25", "city": "Los Angeles"},
        {"name": "Charlie", "age": "35", "city": "New York"}
    ]

    # Write CSV
    write_csv_file(sample_data, "people.csv")

    # Read CSV
    loaded_data = read_csv_file("people.csv")
    print("Loaded data:", loaded_data)

    # Filter data
    if loaded_data:
        ny_people = filter_csv_data(loaded_data, lambda row: row["city"] == "New York")
        print("People in New York:", ny_people)'''

        else:
            return '''def read_text_file(file_path, encoding='utf-8'):
    """Read content from text file."""
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()
        print(f"Successfully read {len(content)} characters from {file_path}")
        return content
    except FileNotFoundError:
        print(f"File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def write_text_file(content, file_path, encoding='utf-8'):
    """Write content to text file."""
    try:
        with open(file_path, 'w', encoding=encoding) as file:
            file.write(content)
        print(f"Successfully wrote content to {file_path}")
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

def append_to_file(content, file_path, encoding='utf-8'):
    """Append content to existing file."""
    try:
        with open(file_path, 'a', encoding=encoding) as file:
            file.write(content)
        print(f"Successfully appended content to {file_path}")
        return True
    except Exception as e:
        print(f"Error appending to file: {e}")
        return False

def process_file_lines(file_path, process_func):
    """Process file line by line."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            processed_lines = []
            for line_num, line in enumerate(file, 1):
                processed_line = process_func(line.strip(), line_num)
                if processed_line is not None:
                    processed_lines.append(processed_line)
        return processed_lines
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Write sample content
    sample_text = "Hello, World!\\nThis is a test file.\\nLine 3 content."
    write_text_file(sample_text, "sample.txt")

    # Read content
    content = read_text_file("sample.txt")
    print("File content:", content)

    # Process lines
    def uppercase_line(line, line_num):
        return f"Line {line_num}: {line.upper()}"

    processed = process_file_lines("sample.txt", uppercase_line)
    print("Processed lines:", processed)'''

    def _get_algorithm_template(self, description: str) -> str:
        """Get algorithm templates"""
        if "sort" in description:
            return '''def bubble_sort(arr):
    """Bubble sort algorithm."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def quick_sort(arr):
    """Quick sort algorithm."""
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)

def merge_sort(arr):
    """Merge sort algorithm."""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    """Merge two sorted arrays."""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Example usage
if __name__ == "__main__":
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", test_array)

    # Test different sorting algorithms
    print("Bubble sort:", bubble_sort(test_array.copy()))
    print("Quick sort:", quick_sort(test_array.copy()))
    print("Merge sort:", merge_sort(test_array.copy()))'''

        elif "search" in description:
            return '''def linear_search(arr, target):
    """Linear search algorithm."""
    for i, value in enumerate(arr):
        if value == target:
            return i
    return -1

def binary_search(arr, target):
    """Binary search algorithm (requires sorted array)."""
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

def binary_search_recursive(arr, target, left=0, right=None):
    """Recursive binary search."""
    if right is None:
        right = len(arr) - 1

    if left > right:
        return -1

    mid = (left + right) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

def find_all_occurrences(arr, target):
    """Find all occurrences of target in array."""
    indices = []
    for i, value in enumerate(arr):
        if value == target:
            indices.append(i)
    return indices

# Example usage
if __name__ == "__main__":
    # Linear search example
    numbers = [64, 34, 25, 12, 22, 11, 90]
    target = 22

    linear_result = linear_search(numbers, target)
    print(f"Linear search for {target}: index {linear_result}")

    # Binary search example (requires sorted array)
    sorted_numbers = sorted(numbers)
    binary_result = binary_search(sorted_numbers, target)
    print(f"Binary search for {target}: index {binary_result}")

    # Find all occurrences
    numbers_with_duplicates = [1, 2, 3, 2, 4, 2, 5]
    all_twos = find_all_occurrences(numbers_with_duplicates, 2)
    print(f"All occurrences of 2: {all_twos}")'''

        else:
            return '''def fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def fibonacci_iterative(n):
    """Calculate nth Fibonacci number iteratively."""
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def factorial(n):
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def factorial_iterative(n):
    """Calculate factorial iteratively."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def gcd(a, b):
    """Calculate Greatest Common Divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Calculate Least Common Multiple."""
    return abs(a * b) // gcd(a, b)

# Example usage
if __name__ == "__main__":
    print("Fibonacci(10):", fibonacci(10))
    print("Fibonacci iterative(10):", fibonacci_iterative(10))
    print("Factorial(5):", factorial(5))
    print("Factorial iterative(5):", factorial_iterative(5))
    print("GCD(48, 18):", gcd(48, 18))
    print("LCM(48, 18):", lcm(48, 18))'''

    def _get_oop_template(self, description: str) -> str:
        """Get object-oriented programming templates"""
        if "class" in description and "inherit" in description:
            return '''class Animal:
    """Base class for animals."""

    def __init__(self, name, species):
        self.name = name
        self.species = species
        self.is_alive = True

    def speak(self):
        """Make a sound - to be overridden by subclasses."""
        pass

    def move(self):
        """Move - to be overridden by subclasses."""
        print(f"{self.name} is moving")

    def __str__(self):
        return f"{self.name} ({self.species})"

class Dog(Animal):
    """Dog class inheriting from Animal."""

    def __init__(self, name, breed):
        super().__init__(name, "Dog")
        self.breed = breed

    def speak(self):
        return f"{self.name} says Woof!"

    def fetch(self):
        return f"{self.name} is fetching the ball"

class Cat(Animal):
    """Cat class inheriting from Animal."""

    def __init__(self, name, color):
        super().__init__(name, "Cat")
        self.color = color

    def speak(self):
        return f"{self.name} says Meow!"

    def climb(self):
        return f"{self.name} is climbing a tree"

# Example usage
if __name__ == "__main__":
    dog = Dog("Buddy", "Golden Retriever")
    cat = Cat("Whiskers", "Orange")

    print(dog)
    print(dog.speak())
    print(dog.fetch())

    print(cat)
    print(cat.speak())
    print(cat.climb())'''

        else:
            return '''class Person:
    """A simple Person class demonstrating OOP concepts."""

    # Class variable
    species = "Homo sapiens"

    def __init__(self, name, age, email=None):
        """Initialize a Person instance."""
        self.name = name
        self.age = age
        self.email = email
        self._id = id(self)  # Private attribute

    def introduce(self):
        """Introduce the person."""
        intro = f"Hi, I'm {self.name} and I'm {self.age} years old."
        if self.email:
            intro += f" You can reach me at {self.email}."
        return intro

    def have_birthday(self):
        """Increment age by 1."""
        self.age += 1
        return f"Happy birthday {self.name}! You are now {self.age}."

    def update_email(self, new_email):
        """Update email address."""
        old_email = self.email
        self.email = new_email
        return f"Email updated from {old_email} to {new_email}"

    @property
    def is_adult(self):
        """Check if person is an adult."""
        return self.age >= 18

    @classmethod
    def from_string(cls, person_str):
        """Create Person from string format 'name,age,email'."""
        parts = person_str.split(',')
        name = parts[0].strip()
        age = int(parts[1].strip())
        email = parts[2].strip() if len(parts) > 2 else None
        return cls(name, age, email)

    @staticmethod
    def is_valid_age(age):
        """Check if age is valid."""
        return 0 <= age <= 150

    def __str__(self):
        return f"Person(name='{self.name}', age={self.age})"

    def __repr__(self):
        return f"Person('{self.name}', {self.age}, '{self.email}')"

# Example usage
if __name__ == "__main__":
    # Create person instances
    person1 = Person("Alice", 25, "alice@email.com")
    person2 = Person.from_string("Bob,30,bob@email.com")

    print(person1.introduce())
    print(person2.introduce())

    # Use methods
    print(person1.have_birthday())
    print(f"Is {person1.name} an adult? {person1.is_adult}")

    # Static method
    print(f"Is age 25 valid? {Person.is_valid_age(25)}")
    print(f"Is age -5 valid? {Person.is_valid_age(-5)}")'''

    def _get_database_template(self, description: str) -> str:
        """Get database templates"""
        if "sqlite" in description:
            return '''import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Any

class DatabaseManager:
    """SQLite database manager."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()

    def init_database(self):
        """Initialize database with tables."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    age INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def create_user(self, name: str, email: str, age: int = None) -> int:
        """Create a new user."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
                (name, email, age)
            )
            conn.commit()
            return cursor.lastrowid

    def get_user(self, user_id: int) -> Dict[str, Any]:
        """Get user by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM users ORDER BY name")
            return [dict(row) for row in cursor.fetchall()]

    def update_user(self, user_id: int, **kwargs) -> bool:
        """Update user fields."""
        if not kwargs:
            return False

        fields = ", ".join(f"{key} = ?" for key in kwargs.keys())
        values = list(kwargs.values()) + [user_id]

        with self.get_connection() as conn:
            cursor = conn.execute(
                f"UPDATE users SET {fields} WHERE id = ?", values
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_user(self, user_id: int) -> bool:
        """Delete user by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0

# Example usage
if __name__ == "__main__":
    db = DatabaseManager("users.db")

    # Create users
    user1_id = db.create_user("Alice", "alice@email.com", 25)
    user2_id = db.create_user("Bob", "bob@email.com", 30)

    print(f"Created users with IDs: {user1_id}, {user2_id}")

    # Get user
    user = db.get_user(user1_id)
    print(f"User: {user}")

    # Get all users
    all_users = db.get_all_users()
    print(f"All users: {all_users}")

    # Update user
    db.update_user(user1_id, age=26, name="Alice Smith")

    # Delete user
    db.delete_user(user2_id)
    print("User deleted")'''

        else:
            return '''# Generic database operations example
class SimpleDatabase:
    """Simple in-memory database simulation."""

    def __init__(self):
        self.tables = {}
        self.next_id = 1

    def create_table(self, table_name: str, columns: list):
        """Create a new table."""
        self.tables[table_name] = {
            'columns': columns,
            'data': [],
            'indexes': {}
        }
        return f"Table '{table_name}' created with columns: {columns}"

    def insert(self, table_name: str, record: dict):
        """Insert a record into table."""
        if table_name not in self.tables:
            return f"Table '{table_name}' does not exist"

        # Add auto-increment ID
        record['id'] = self.next_id
        self.next_id += 1

        self.tables[table_name]['data'].append(record)
        return f"Record inserted with ID: {record['id']}"

    def select(self, table_name: str, condition=None):
        """Select records from table."""
        if table_name not in self.tables:
            return []

        data = self.tables[table_name]['data']

        if condition is None:
            return data

        return [record for record in data if condition(record)]

    def update(self, table_name: str, record_id: int, updates: dict):
        """Update a record."""
        if table_name not in self.tables:
            return False

        for record in self.tables[table_name]['data']:
            if record.get('id') == record_id:
                record.update(updates)
                return True
        return False

    def delete(self, table_name: str, record_id: int):
        """Delete a record."""
        if table_name not in self.tables:
            return False

        data = self.tables[table_name]['data']
        for i, record in enumerate(data):
            if record.get('id') == record_id:
                del data[i]
                return True
        return False

    def show_table(self, table_name: str):
        """Display table contents."""
        if table_name not in self.tables:
            return f"Table '{table_name}' does not exist"

        table = self.tables[table_name]
        print(f"\\nTable: {table_name}")
        print(f"Columns: {table['columns']}")
        print("Data:")
        for record in table['data']:
            print(f"  {record}")

# Example usage
if __name__ == "__main__":
    db = SimpleDatabase()

    # Create table
    db.create_table('users', ['name', 'email', 'age'])

    # Insert records
    db.insert('users', {'name': 'Alice', 'email': 'alice@email.com', 'age': 25})
    db.insert('users', {'name': 'Bob', 'email': 'bob@email.com', 'age': 30})

    # Select all
    all_users = db.select('users')
    print("All users:", all_users)

    # Select with condition
    adults = db.select('users', lambda r: r['age'] >= 25)
    print("Adults:", adults)

    # Update
    db.update('users', 1, {'age': 26})

    # Show table
    db.show_table('users')'''

    def _get_testing_template(self, description: str) -> str:
        """Get testing templates"""
        return '''import unittest
from unittest.mock import Mock, patch

# Example class to test
class Calculator:
    """Simple calculator class for testing demonstration."""

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def power(self, base, exponent):
        return base ** exponent

class TestCalculator(unittest.TestCase):
    """Test cases for Calculator class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.calc = Calculator()

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_add(self):
        """Test addition operation."""
        self.assertEqual(self.calc.add(2, 3), 5)
        self.assertEqual(self.calc.add(-1, 1), 0)
        self.assertEqual(self.calc.add(0, 0), 0)

    def test_subtract(self):
        """Test subtraction operation."""
        self.assertEqual(self.calc.subtract(5, 3), 2)
        self.assertEqual(self.calc.subtract(0, 5), -5)
        self.assertEqual(self.calc.subtract(-2, -3), 1)

    def test_multiply(self):
        """Test multiplication operation."""
        self.assertEqual(self.calc.multiply(3, 4), 12)
        self.assertEqual(self.calc.multiply(-2, 3), -6)
        self.assertEqual(self.calc.multiply(0, 100), 0)

    def test_divide(self):
        """Test division operation."""
        self.assertEqual(self.calc.divide(10, 2), 5)
        self.assertEqual(self.calc.divide(7, 2), 3.5)

        # Test exception
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)

    def test_power(self):
        """Test power operation."""
        self.assertEqual(self.calc.power(2, 3), 8)
        self.assertEqual(self.calc.power(5, 0), 1)
        self.assertEqual(self.calc.power(2, -1), 0.5)

    @patch('builtins.print')
    def test_with_mock(self, mock_print):
        """Example of using mocks in tests."""
        result = self.calc.add(2, 3)
        print(f"Result: {result}")
        mock_print.assert_called_once_with("Result: 5")

class TestWithMocks(unittest.TestCase):
    """Examples of testing with mocks."""

    def test_mock_example(self):
        """Example of creating and using mocks."""
        # Create a mock object
        mock_service = Mock()

        # Configure mock return value
        mock_service.get_data.return_value = {"key": "value"}

        # Use the mock
        result = mock_service.get_data()

        # Assert the mock was called and returned expected value
        mock_service.get_data.assert_called_once()
        self.assertEqual(result, {"key": "value"})

    @patch('requests.get')
    def test_api_call(self, mock_get):
        """Example of mocking external API calls."""
        # Configure mock response
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Function that makes API call (example)
        def fetch_data():
            import requests
            response = requests.get("https://api.example.com/data")
            return response.json()

        # Test the function
        result = fetch_data()

        # Assertions
        mock_get.assert_called_once_with("https://api.example.com/data")
        self.assertEqual(result, {"status": "success"})

if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)'''

    def _get_ml_template(self, description: str) -> str:
        """Get machine learning templates"""
        if "sklearn" in description or "scikit" in description:
            return '''import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MLPipeline:
    """Machine Learning Pipeline for classification tasks."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def load_data(self, file_path):
        """Load data from CSV file."""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_data(self, target_column, test_size=0.2):
        """Preprocess data for training."""
        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        # Handle categorical variables (simple encoding)
        X = pd.get_dummies(X, drop_first=True)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"Training set: {self.X_train_scaled.shape}")
        print(f"Test set: {self.X_test_scaled.shape}")

        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def train_model(self, model_type='logistic'):
        """Train the machine learning model."""
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Unsupported model type")

        # Train the model
        self.model.fit(self.X_train_scaled, self.y_train)
        self.is_trained = True

        print(f"Model trained: {type(self.model).__name__}")
        return self.model

    def evaluate_model(self):
        """Evaluate the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        # Make predictions
        y_pred = self.model.predict(self.X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print("\\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

        return accuracy, y_pred

    def predict(self, new_data):
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        # Scale new data
        new_data_scaled = self.scaler.transform(new_data)

        # Make predictions
        predictions = self.model.predict(new_data_scaled)
        probabilities = self.model.predict_proba(new_data_scaled)

        return predictions, probabilities

# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = MLPipeline()

    # Example with synthetic data
    from sklearn.datasets import make_classification

    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    # Save to CSV for demonstration
    df.to_csv('sample_data.csv', index=False)

    # Load and process data
    pipeline.load_data('sample_data.csv')
    pipeline.preprocess_data('target')

    # Train and evaluate model
    pipeline.train_model('random_forest')
    accuracy, predictions = pipeline.evaluate_model()

    print(f"\\nModel training completed with accuracy: {accuracy:.4f}")'''

        else:
            return '''import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    """Simple neural network implementation from scratch."""

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        return x * (1 - x)

    def forward(self, X):
        """Forward propagation."""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        """Backward propagation."""
        m = X.shape[0]

        # Calculate gradients
        dZ2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        """Train the neural network."""
        costs = []

        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)

            # Calculate cost
            cost = np.mean((output - y) ** 2)
            costs.append(cost)

            # Backward propagation
            dW1, db1, dW2, db2 = self.backward(X, y, output)

            # Update weights and biases
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.6f}")

        return costs

    def predict(self, X):
        """Make predictions."""
        return self.forward(X)

# Example usage
if __name__ == "__main__":
    # Generate sample data (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Create and train network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    costs = nn.train(X, y, epochs=5000, learning_rate=1.0)

    # Make predictions
    predictions = nn.predict(X)

    print("\\nPredictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")

    # Plot training cost
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title('Training Cost Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()'''

    def _get_web_template(self, description: str) -> str:
        """Get web development templates"""
        return '''from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
from pathlib import Path

class WebHandler(BaseHTTPRequestHandler):
    """Simple web server handler."""

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path == '/':
            self.serve_html_page()
        elif path == '/api/data':
            self.serve_json_data()
        elif path.startswith('/static/'):
            self.serve_static_file(path)
        else:
            self.send_404()

    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/api/submit':
            self.handle_form_submission()
        else:
            self.send_404()

    def serve_html_page(self):
        """Serve the main HTML page."""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Web App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { color: #333; }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input, textarea {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        #result {
            margin-top: 20px;
            padding: 10px;
            background-color: #e9ecef;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Simple Web Application</h1>

        <form id="dataForm">
            <div class="form-group">
                <label for="name">Name:</label>
                <input type="text" id="name" name="name" required>
            </div>

            <div class="form-group">
                <label for="email">Email:</label>
                <input type="email" id="email" name="email" required>
            </div>

            <div class="form-group">
                <label for="message">Message:</label>
                <textarea id="message" name="message" rows="4"></textarea>
            </div>

            <button type="submit">Submit</button>
        </form>

        <div id="result"></div>

        <h2>API Data</h2>
        <button onclick="loadData()">Load Data</button>
        <div id="apiData"></div>
    </div>

    <script>
        // Handle form submission
        document.getElementById('dataForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const formData = new FormData(this);
            const data = Object.fromEntries(formData);

            try {
                const response = await fetch('/api/submit', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                document.getElementById('result').innerHTML =
                    '<h3>Submission Result:</h3><pre>' + JSON.stringify(result, null, 2) + '</pre>';
            } catch (error) {
                document.getElementById('result').innerHTML =
                    '<h3>Error:</h3><p>' + error.message + '</p>';
            }
        });

        // Load API data
        async function loadData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                document.getElementById('apiData').innerHTML =
                    '<h3>API Data:</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
            } catch (error) {
                document.getElementById('apiData').innerHTML =
                    '<h3>Error:</h3><p>' + error.message + '</p>';
            }
        }
    </script>
</body>
</html>"""

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())

    def serve_json_data(self):
        """Serve JSON data."""
        data = {
            "message": "Hello from the API!",
            "timestamp": "2024-01-01T12:00:00Z",
            "data": [
                {"id": 1, "name": "Item 1", "value": 100},
                {"id": 2, "name": "Item 2", "value": 200},
                {"id": 3, "name": "Item 3", "value": 300}
            ]
        }

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def handle_form_submission(self):
        """Handle form data submission."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode('utf-8'))

            # Process the submitted data
            response = {
                "status": "success",
                "message": f"Thank you, {data.get('name', 'User')}!",
                "received_data": data
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON data")

    def serve_static_file(self, path):
        """Serve static files."""
        # Simple static file serving (for demonstration)
        self.send_404()

    def send_404(self):
        """Send 404 error."""
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<h1>404 - Page Not Found</h1>')

def run_server(port=8080):
    """Run the web server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, WebHandler)
    print(f"Server running on http://localhost:{port}")
    print("Press Ctrl+C to stop the server")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\\nServer stopped")
        httpd.server_close()

if __name__ == "__main__":
    run_server()'''

    def _get_concurrency_template(self, description: str) -> str:
        """Get concurrency templates"""
        if "async" in description or "asyncio" in description:
            return '''import asyncio
import aiohttp
import time
from typing import List

async def fetch_url(session: aiohttp.ClientSession, url: str) -> dict:
    """Fetch a single URL asynchronously."""
    try:
        async with session.get(url) as response:
            return {
                "url": url,
                "status": response.status,
                "content_length": len(await response.text()),
                "success": True
            }
    except Exception as e:
        return {
            "url": url,
            "error": str(e),
            "success": False
        }

async def fetch_multiple_urls(urls: List[str]) -> List[dict]:
    """Fetch multiple URLs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

async def process_data_async(data_list: List[str]) -> List[str]:
    """Process data asynchronously."""
    async def process_item(item: str) -> str:
        # Simulate async processing
        await asyncio.sleep(0.1)
        return f"Processed: {item.upper()}"

    tasks = [process_item(item) for item in data_list]
    results = await asyncio.gather(*tasks)
    return results

class AsyncWorker:
    """Async worker class for background tasks."""

    def __init__(self):
        self.is_running = False
        self.tasks = []

    async def add_task(self, coro):
        """Add a coroutine task."""
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task

    async def background_worker(self):
        """Background worker that runs continuously."""
        self.is_running = True
        counter = 0

        while self.is_running:
            print(f"Background worker tick: {counter}")
            counter += 1
            await asyncio.sleep(2)

    async def stop_worker(self):
        """Stop the background worker."""
        self.is_running = False

        # Cancel all pending tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete or be cancelled
        await asyncio.gather(*self.tasks, return_exceptions=True)

async def producer(queue: asyncio.Queue, items: List[str]):
    """Producer that adds items to queue."""
    for item in items:
        await queue.put(item)
        print(f"Produced: {item}")
        await asyncio.sleep(0.1)

    # Signal completion
    await queue.put(None)

async def consumer(queue: asyncio.Queue, consumer_id: int):
    """Consumer that processes items from queue."""
    while True:
        item = await queue.get()

        if item is None:
            # Signal other consumers to stop
            await queue.put(None)
            break

        # Process item
        await asyncio.sleep(0.2)  # Simulate processing time
        print(f"Consumer {consumer_id} processed: {item}")
        queue.task_done()

async def main():
    """Main async function demonstrating various patterns."""
    print("=== Async HTTP Requests ===")
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/json"
    ]

    start_time = time.time()
    results = await fetch_multiple_urls(urls)
    end_time = time.time()

    print(f"Fetched {len(results)} URLs in {end_time - start_time:.2f} seconds")
    for result in results:
        if result["success"]:
            print(f"✓ {result['url']} - Status: {result['status']}")
        else:
            print(f"✗ {result['url']} - Error: {result['error']}")

    print("\\n=== Async Data Processing ===")
    data = ["item1", "item2", "item3", "item4", "item5"]
    processed = await process_data_async(data)
    print("Processed items:", processed)

    print("\\n=== Producer-Consumer Pattern ===")
    queue = asyncio.Queue(maxsize=3)
    items = ["task1", "task2", "task3", "task4", "task5"]

    # Create producer and consumers
    producer_task = asyncio.create_task(producer(queue, items))
    consumer_tasks = [
        asyncio.create_task(consumer(queue, i))
        for i in range(2)
    ]

    # Wait for completion
    await producer_task
    await asyncio.gather(*consumer_tasks)

    print("\\n=== Background Worker ===")
    worker = AsyncWorker()

    # Start background worker
    worker_task = await worker.add_task(worker.background_worker())

    # Let it run for a few seconds
    await asyncio.sleep(5)

    # Stop worker
    await worker.stop_worker()
    print("Background worker stopped")

if __name__ == "__main__":
    asyncio.run(main())'''

        else:
            return '''import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable

class ThreadSafeCounter:
    """Thread-safe counter using locks."""

    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._value += 1

    def decrement(self):
        with self._lock:
            self._value -= 1

    @property
    def value(self):
        with self._lock:
            return self._value

def worker_function(worker_id: int, work_queue: queue.Queue, results: List):
    """Worker function that processes items from queue."""
    while True:
        try:
            item = work_queue.get(timeout=1)
            if item is None:  # Poison pill to stop worker
                break

            # Simulate work
            result = f"Worker {worker_id} processed {item}"
            time.sleep(0.1)

            results.append(result)
            work_queue.task_done()

        except queue.Empty:
            break

def parallel_processing_example():
    """Example of parallel processing with threads."""
    print("=== Thread Pool Example ===")

    def process_item(item):
        """Function to process a single item."""
        time.sleep(0.1)  # Simulate work
        return f"Processed: {item ** 2}"

    items = list(range(1, 11))

    # Using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(process_item, item): item for item in items}

        results = []
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed: {result}")
            except Exception as e:
                print(f"Item {item} generated an exception: {e}")

    return results

def producer_consumer_example():
    """Producer-consumer pattern with threads."""
    print("\\n=== Producer-Consumer Example ===")

    work_queue = queue.Queue(maxsize=5)
    results = []

    # Create and start worker threads
    workers = []
    for i in range(3):
        worker = threading.Thread(
            target=worker_function,
            args=(i, work_queue, results)
        )
        worker.start()
        workers.append(worker)

    # Producer: add work items
    for i in range(10):
        work_queue.put(f"task_{i}")
        print(f"Added task_{i} to queue")

    # Wait for all tasks to be processed
    work_queue.join()

    # Stop workers by sending poison pills
    for _ in workers:
        work_queue.put(None)

    # Wait for all workers to finish
    for worker in workers:
        worker.join()

    print(f"All work completed. Results: {len(results)} items")
    for result in results:
        print(f"  {result}")

class ThreadedDownloader:
    """Example of threaded file/data downloader."""

    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.counter = ThreadSafeCounter()

    def download_item(self, url: str) -> dict:
        """Simulate downloading an item."""
        self.counter.increment()

        # Simulate download time
        time.sleep(0.2)

        result = {
            "url": url,
            "status": "success",
            "size": len(url) * 100,  # Fake size
            "thread_id": threading.current_thread().ident
        }

        print(f"Downloaded: {url} (Thread: {result['thread_id']})")
        return result

    def download_all(self, urls: List[str]) -> List[dict]:
        """Download all URLs using thread pool."""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self.download_item, url): url
                for url in urls
            }

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Failed to download {url}: {e}")

        return results

def main():
    """Main function demonstrating threading concepts."""
    # Parallel processing
    parallel_results = parallel_processing_example()

    # Producer-consumer pattern
    producer_consumer_example()

    # Threaded downloader
    print("\\n=== Threaded Downloader Example ===")
    downloader = ThreadedDownloader(max_workers=3)

    urls = [
        "https://example.com/file1.txt",
        "https://example.com/file2.txt",
        "https://example.com/file3.txt",
        "https://example.com/file4.txt",
        "https://example.com/file5.txt"
    ]

    start_time = time.time()
    download_results = downloader.download_all(urls)
    end_time = time.time()

    print(f"\\nDownloaded {len(download_results)} files in {end_time - start_time:.2f} seconds")
    print(f"Total counter value: {downloader.counter.value}")

if __name__ == "__main__":
    main()'''

    def _get_fallback_code(self, description: str, language: ProgrammingLanguage) -> str:
        """Get fallback code when generation fails"""
        if language == ProgrammingLanguage.PYTHON:
            return f'''# Generated Python code for: {description}

def generated_function():
    """
    This function was generated based on your request: {description}
    Please modify it according to your specific needs.
    """
    # TODO: Implement your logic here
    print("Function generated successfully!")
    return True

# Example usage
if __name__ == "__main__":
    result = generated_function()
    print(f"Result: {{result}}")'''

        elif language == ProgrammingLanguage.JAVASCRIPT:
            return f'''// Generated JavaScript code for: {description}

function generatedFunction() {{
    /*
     * This function was generated based on your request: {description}
     * Please modify it according to your specific needs.
     */
    // TODO: Implement your logic here
    console.log("Function generated successfully!");
    return true;
}}

// Example usage
const result = generatedFunction();
console.log(`Result: ${{result}}`);'''

        else:
            return f"// Generated {language.value} code for: {description}\n// TODO: Implement your logic here"

    def _validate_code(self, code: str, language: ProgrammingLanguage) -> Optional[str]:
        """Validate generated code for syntax errors"""
        try:
            if language == ProgrammingLanguage.PYTHON:
                ast.parse(code)
                return "Python syntax is valid"
            # Add validation for other languages as needed
            return "Code appears to be well-formed"
        except SyntaxError as e:
            return f"Syntax error found: {str(e)}"
        except Exception:
            return None
    
    def _analyze_code_issues(self, code: str, language: ProgrammingLanguage) -> List[str]:
        """Analyze code for potential issues"""
        issues = []
        
        if language == ProgrammingLanguage.PYTHON:
            # Check for common Python issues
            if 'except:' in code:
                issues.append("Bare except clause found - consider catching specific exceptions")
            if 'eval(' in code:
                issues.append("Use of eval() detected - potential security risk")
            if re.search(r'print\s*\(', code):
                issues.append("Print statements found - consider using logging instead")
        
        # Add more language-specific checks
        return issues
    
    def _generate_code_suggestions(self, code: str, language: ProgrammingLanguage) -> List[str]:
        """Generate suggestions for code improvement"""
        suggestions = []
        
        # Generic suggestions
        if len(code.split('\n')) > 50:
            suggestions.append("Consider breaking this into smaller functions")
        
        if language == ProgrammingLanguage.PYTHON:
            if 'def ' in code and '"""' not in code:
                suggestions.append("Add docstrings to functions for better documentation")
        
        return suggestions
    
    def _rate_code_quality(self, code: str, language: ProgrammingLanguage) -> int:
        """Rate code quality on a scale of 1-10"""
        score = 7  # Base score
        
        # Adjust based on various factors
        lines = code.split('\n')
        
        # Penalize very long functions
        if len(lines) > 100:
            score -= 2
        
        # Reward documentation
        if '"""' in code or '//' in code or '/*' in code:
            score += 1
        
        # Penalize obvious issues
        if 'TODO' in code or 'FIXME' in code:
            score -= 1
        
        return max(1, min(10, score))
    
    def _get_default_test_framework(self, language: ProgrammingLanguage) -> str:
        """Get default test framework for language"""
        frameworks = {
            ProgrammingLanguage.PYTHON: "pytest",
            ProgrammingLanguage.JAVASCRIPT: "jest",
            ProgrammingLanguage.JAVA: "junit",
            ProgrammingLanguage.GO: "testing"
        }
        return frameworks.get(language, "standard")
    
    # Placeholder methods for more complex functionality
    async def _improve_code(self, code: str, issues: List[str], suggestions: List[str], 
                          language: ProgrammingLanguage) -> str:
        """Generate improved version of code"""
        # This would use LLM to improve the code based on issues and suggestions
        return code  # Placeholder
    
    def _analyze_code_structure(self, code: str, language: ProgrammingLanguage) -> str:
        """Analyze code structure"""
        return "Code structure analysis would go here"  # Placeholder
    
    def _find_common_bugs(self, code: str, language: ProgrammingLanguage) -> List[str]:
        """Find common bugs in code"""
        return []  # Placeholder
    
    def _analyze_error_message(self, error: str, code: str, language: ProgrammingLanguage) -> str:
        """Analyze error message"""
        return f"Error analysis for: {error}"  # Placeholder
    
    def _generate_debug_suggestions(self, code: str, language: ProgrammingLanguage, error: str) -> List[str]:
        """Generate debugging suggestions"""
        return ["Check variable names", "Verify function calls"]  # Placeholder
    
    def _attempt_code_fix(self, code: str, language: ProgrammingLanguage, bugs: List[str], error: str) -> str:
        """Attempt to fix code"""
        return code  # Placeholder
    
    def _find_performance_optimizations(self, code: str, language: ProgrammingLanguage) -> List[str]:
        """Find performance optimization opportunities"""
        return []  # Placeholder
    
    def _apply_performance_optimizations(self, code: str, optimizations: List[str], 
                                       language: ProgrammingLanguage) -> str:
        """Apply performance optimizations"""
        return code  # Placeholder
    
    def _find_readability_improvements(self, code: str, language: ProgrammingLanguage) -> List[str]:
        """Find readability improvements"""
        return []  # Placeholder
    
    def _apply_readability_improvements(self, code: str, improvements: List[str], 
                                      language: ProgrammingLanguage) -> str:
        """Apply readability improvements"""
        return code  # Placeholder
    
    def _generate_conversion_notes(self, source: ProgrammingLanguage, target: ProgrammingLanguage) -> str:
        """Generate notes about code conversion"""
        return f"Converted from {source.value} to {target.value}"  # Placeholder
    
    def _identify_test_cases(self, code: str, language: ProgrammingLanguage) -> List[str]:
        """Identify test cases for code"""
        return ["Test basic functionality", "Test edge cases"]  # Placeholder
    
    def _generate_test_code(self, code: str, test_cases: List[str], language: ProgrammingLanguage, 
                          framework: str) -> str:
        """Generate test code"""
        return f"# Test code for {framework} would go here"  # Placeholder
    
    def _get_test_execution_instructions(self, language: ProgrammingLanguage, framework: str) -> str:
        """Get test execution instructions"""
        return f"Run tests with: {framework}"  # Placeholder
    
    def _extract_functions(self, code: str, language: ProgrammingLanguage) -> List[str]:
        """Extract function names from code"""
        return []  # Placeholder
    
    def _extract_classes(self, code: str, language: ProgrammingLanguage) -> List[str]:
        """Extract class names from code"""
        return []  # Placeholder
    
    def _create_documentation(self, code: str, functions: List[str], classes: List[str], 
                            language: ProgrammingLanguage, style: str) -> str:
        """Create documentation"""
        return "Documentation would go here"  # Placeholder
    
    def _generate_usage_example(self, code: str, language: ProgrammingLanguage) -> Optional[str]:
        """Generate usage example for code"""
        return None  # Placeholder
