# ai-pose-trainer

AI-based Real-time Fitness Posture Correction and Coaching System

## Project Structure

The project is organized as follows:

- `data/`: Directory for storing datasets
- `demo/`: Scripts for running the demo
- `script/`: Directory for experiment and prototype scripts
- `src/`: Directory for source code
  - `data/`: Code for data processing
  - `exercise/`: Code for each exercise modules
  - `model/`: Code for model implementations
  - `utils/`: Directory for utility functions

## Installation

### Prerequisites

- Python 3.10
- uv (for dependency management and virtual environment)

### Setup

#### Option 1: Using uv (Recommended)

1. Install uv if you don't have it already:

   ```
   pip install uv
   ```

2. Clone the repository:

   ```
   git clone https://github.com/AI-Contact/ai-pose-trainer.git
   cd ai-pose-trainer
   ```

3. Create a virtual environment and install dependencies using uv:

   ```
   uv venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

4. Install PyTorch with CUDA support:

   ```
   uv pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

5. Install dependencies:

   ```
   uv pip install -r requirements.txt
   ```

#### Option 2: Using pipenv (Legacy)

1. Install dependencies using pipenv:

   ```
   pipenv install
   ```

2. Activate the virtual environment:
   ```
   pipenv shell
   ```
