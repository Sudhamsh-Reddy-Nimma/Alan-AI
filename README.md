Alan AI: Hybrid RL + Cognitive Architecture (v2) 🧠🤖

  Alan is an autonomous stick-figure agent built with a custom cognitive
  architecture that sits on top of a Reinforcement Learning policy. Unlike
  traditional agents that only maximize a reward signal, Alan possesses a
  "biological" internal state including emotions, fatigue, and a circadian rhythm
  that influences its decision-making process.

  !Alan AI Gameplay
  (https://via.placeholder.com/800x450.png?text=Alan+AI+Stick+Figure+Sandbox)
  (Replace this with a screenshot or GIF from your screenshots/ folder later!)

  🚀 Key Features

  🧠 1. Cognitive Pipeline
  Alan doesn't just react; he thinks. The decision-making process follows a
  structured loop:
  Perception → Interpretation → Thought → Intent Selection → RL Policy → Action

  🎭 2. Multi-Channel Emotion System
  Alan experiences six distinct emotional channels that influence his "Intent"
  selection:
   * Happiness & Surprise: Increases curiosity and playfulness.
   * Fear: Triggered by heights or falls; promotes exploration of safe zones.
   * Anger & Sadness: Arises from failure or exhaustion; leads to behavioral
     shifts.

  ☀️ 3. Biological Constraints
   * Circadian Rhythm: A natural 5-minute day/night cycle. Nightfall increases
     sleep drive, affecting Alan's energy and mood.
   * Fatigue & Welfare: Dynamic levels of Energy, Hunger, and Fatigue. High fatigue
     reduces movement efficiency and decision-making frequency.
   * Farming & Survival: Alan can gather seeds, plant crops, and harvest food to
     manage his hunger levels.

  📈 4. RL & Memory
   * PPO-Compatible Policy: Designed to be trained with Proximal Policy
     Optimization.
   * Recency-Weighted Memory: Alan reflects on recent experiences, allowing his
     personality traits (Confidence, Curiosity, etc.) to evolve over time.

  ---

  🎮 Controls

  You can toggle between Autonomous (AI) and Manual (Player) modes.

  │ Key        │ Action                            │
  ├────────────┼───────────────────────────────────┤
  │ A          │ Toggle Autonomous Brain (AI Mode) │
  │ ← / →      │ Move Left / Right                 │
  │ ↑ / Space  │ Jump                              │
  │ C          │ Crouch                            │
  │ G / T      │ Grab / Throw Object               │
  │ S / Z      │ Sit / Sleep                       │
  │ 1, 2, 3, 4 │ Simulation Speed (0.5x to 4x)     │
  │ D          │ Toggle Debug Overlay              │
  │ P          │ Pause / Resume                    │
 
  ---

  🛠️ Tech Stack
   * Language: Python 3.x
   * Physics Engine: Pymunk
   * Graphics: Pygame
   * AI/RL: OpenAI Gymnasium, Stable-Baselines3 (Optional)
   * LLM Integration: Google Gemini API (Optional Advisor)

  ---

  📦 Installation & Usage

   1. Clone the repo:

   1    git clone https://github.com/Sudhamsh-Reddy-Nimma/Alan-AI.git
   2    cd Alan-AI

   2. Install dependencies:

   1    pip install pygame pymunk python-dotenv gymnasium

   3. Run the Sandbox:
   1    python game.py
