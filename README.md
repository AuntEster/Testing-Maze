
# Fresh Maze Solver Rewrite

This is a clean rewrite of the maze solver that keeps:
- OpenCV template matching for hazard detection
- the working live visualization style
- the working rotating death-pit logic

Everything else is rebuilt around the project API.

## Files
- `core.py` — action enum, turn result, template matching, maze loading, environment, evaluator
- `agent.py` — fresh exploration/search agent with small Q-learning fallback
- `visualize.py` — live visualizer
- `train.py` — training entrypoint
- `evaluate.py` — evaluation entrypoint
- `run_live.py` — live animated run

## Notes
- Put your hazard templates in a `templates/` folder.
- Template filenames should match the existing labels, like:
  - `death_pit_00_...png`
  - `confusion_00_...png`
  - `teleport_green_00_...png`
  - `teleport_orange_00_...png`
  - `teleport_purple_00_...png`
  - `teleport_red_00_...png`
