.PHONY: setup collect-all collect-squads collect-stats collect-history run-dashboard test clean

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	cp .env.example .env
	mkdir -p logs data/raw data/processed models
	@echo "✅ Setup done. Edit .env to add your API keys."

# ── Data Collection ───────────────────────────────────────────────────────────
collect-squads:
	python3 src/collect/collect_squads.py

collect-history:
	python3 src/collect/collect_rankings_and_history.py

collect-stats:
	python3 src/collect/collect_player_stats.py

collect-all: collect-squads collect-history collect-stats
	@echo "✅ All data collected."

# ── Model ─────────────────────────────────────────────────────────────────────
features:
	python3 src/features/build_features.py

train:
	python3 src/model/train.py

simulate:
	python3 src/model/simulate.py

# ── Dashboard ────────────────────────────────────────────────────────────────
run-dashboard:
	streamlit run src/dashboard/app.py

# ── Dev ───────────────────────────────────────────────────────────────────────
test:
	python3 -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf logs/*.log

# ── Git helpers ───────────────────────────────────────────────────────────────
push:
	git add -A && git commit -m "chore: auto-commit" && git push
