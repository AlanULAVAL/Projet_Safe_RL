# Projet_Safe_RL
Projet du cours IFT-7201 en RL

Ce dépôt contient le code source du projet d’étude sur l'apprentissage par renforcement dans des environnements dynamiques, incluant une évaluation comparative de différentes stratégies d'entraînement sur un environnement modifié de CartPole.

## Objectif

Évaluer la capacité d'agents RL à généraliser dans des conditions dynamiques changeantes (gravité et masse du poteau), en comparant un entraînement standard (PPO) à une version robuste (RARL).

## Organisation des fichiers

### Agents (`src/`)

- `train_ppo.py` — Entraîne un agent PPO simple dans l’environnement standard.
- `train_rarl_rnd_old.py` — Entraîne un agent PPO avec adversaire sur une **plage restreinte** (version RARL simple).
- `train_rarl_rnd.py` — Version avancée avec **curriculum learning** et **early stopping** (version RARL complète et finale).

### Tests (`src/`)

- `test_ppo.py` — Évalue l'agent PPO face à un agent RARL sur une grille d’environnements (gravité × masse), et génère des cartes de chaleur.

### Environnements personnalisés (`src/envs/`)

- `perturbed_env.py` — Définit un environnement CartPole modifié avec gravité et masse personnalisables.
- `rarl_env.py` — Environnement utilisé dans les agents RARL pour permettre à l’adversaire de modifier dynamiquement les paramètres.

### Modèles sauvegardés (`models/`)

- `agent_ppo.zip` — Modèle entraîné avec PPO simple.
- `mid_protagonist_rarl_rnd.zip` — Modèle RARL (version simple, plage restreinte).
- `big_protagonist_rarl_rnd.zip` — Modèle RARL final (curriculum learning, plage étendue).

### Résultats (`results/`)

- `AgentSimple.png` — PPO simple, entraîné sur un seul environnement.
- `MidAgentSafe.png` — RARL avec adversaire (plage restreinte).
- `BigAgentSafe.png` — RARL avec curriculum learning (plage étendue).

Ces images sont intégrées dans le rapport et analysées en détail.

## Dépendances

Installez les bibliothèques nécessaires :

```bash
pip install -r requirements.txt