# VIDUR Codebase Guidelines

## Build & Development Commands
- Format code: `make format` (runs isort and black)
- Lint code: `make lint` (runs black and isort in check mode)
- Run individual lint checks: `make lint/flake8`, `make lint/black`, `make lint/isort`
- Run tests: `pytest path/to/test_file.py::TestClass::test_method` 

## Code Style Guidelines
- Code formatting: Use black and isort (black profile) for consistent formatting
- Imports: Group imports in order - standard library, third-party, local - separated by line breaks
- Types: Use typing module for type annotations (List, Dict, Optional, etc.)
- Classes should follow object-oriented design with clear inheritance hierarchies
- Use dataclasses for configuration and data storage classes
- Registry pattern is used for registration of various components
- Naming: Use snake_case for variables/functions, PascalCase for classes, UPPER_CASE for constants
- Error handling: Follow proper exception handling patterns with specific exceptions
- Documentation: Include docstrings for public methods and classes

This codebase simulates LLM serving systems with scheduling, request handling, and performance metrics collection.

The general architecture of the code is as follows:

  1. Events System: The main driver of the simulation using a priority queue
    - Events are processed chronologically based on their time and priority
    - Each event can generate new events when handled
  2. Event Flow:
    - RequestArrivalEvent → Initial event when requests enter the system
    - GlobalScheduleEvent → Distributes requests across replicas
    - ReplicaScheduleEvent → Forms batches within replicas
    - BatchStageArrivalEvent → Signals batch arrival at pipeline stage
    - ReplicaStageScheduleEvent → Schedules execution at pipeline stage
    - BatchStageEndEvent → Signals completion of a stage
    - BatchEndEvent → Signals complete batch processing
  3. Entity Hierarchy:
    - Cluster contains multiple Replicas
    - Replicas process Batches
    - Batches contain multiple Requests
    - BatchStages represent pipeline execution segments
  4. Scheduler Hierarchy:
    - GlobalScheduler: Distributes requests (Round-robin, Random, LOR)
    - ReplicaScheduler: Manages batching (VLLM, Sarathi, Orca, etc.)
    - ReplicaStageScheduler: Manages pipeline execution
