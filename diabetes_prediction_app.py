"""
Race Day Prediction Web Application

This Flask application provides a web interface for selecting race days and
tracks, sending horse-level prediction requests to a Databricks-served model,
and visualizing results (including a playful race simulation). The structure
mirrors the original diabetes project so deployment, configuration, and
infrastructure stay consistent while the domain logic changes entirely.

Educational Overview:
--------------------
This application demonstrates several important software engineering concepts:
1. Separation of Concerns: Configuration separate from business logic
2. RESTful API Design: Clean endpoints for client-server communication
3. Error Handling: Graceful handling of failures with informative messages
4. Input Validation: Ensuring data quality before processing
5. Documentation: Explaining the "why" behind the code and architecture

Architecture:
------------
- Frontend: HTML/JavaScript that lets users choose a race date/track and view races
- Backend: Flask server that loads schedule data and calls the ML model
- ML Service: Databricks MLflow serving endpoint that returns win probabilities

Author: [Your Name / Team Name]
Last Updated: 2025-01-21
"""

# ============================================================================
# Standard Library Imports
# ============================================================================
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================================
# Third-Party Imports
# ============================================================================
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

# ============================================================================
# Local Imports
# ============================================================================
from config import Config

# ============================================================================
# Application Initialization
# ============================================================================

# Initialize Flask application
app = Flask(__name__)

# Load configuration from the Config class
app.config.from_object(Config)

# Print configuration status on startup for debugging
Config.print_config_status()


# ============================================================================
# Helper Functions
# ============================================================================

def load_race_config() -> Dict[str, Any]:
    """
    Loads the race configuration file that defines dates, tracks, races, and winners.

    Returns:
        Dict: Parsed JSON configuration

    Raises:
        FileNotFoundError: If the configuration file is missing
        ValueError: If the file cannot be parsed as JSON or is missing required keys
    """
    config_path = Path(Config.RACE_CONFIG_FILE)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Race configuration file not found at {config_path.resolve()}"
        )

    try:
        with config_path.open('r', encoding='utf-8') as config_file:
            data = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Race configuration file is not valid JSON: {exc}") from exc

    if 'dates' not in data:
        raise ValueError('Race configuration file must include a "dates" section.')

    return data


def load_winners_config() -> Dict[str, Any]:
    """
    Loads the winners configuration file that maps race_id to winner data.
    """
    config_path = Path(Config.WINNERS_CONFIG_FILE)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Winners configuration file not found at {config_path.resolve()}"
        )

    try:
        with config_path.open('r', encoding='utf-8') as config_file:
            data = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Winners configuration file is not valid JSON: {exc}") from exc

    if 'winners' not in data:
        raise ValueError('Winners configuration file must include a "winners" section.')

    return data


def sortable_number(value: Any) -> tuple[int, Any]:
    """
    Produces a stable, comparable key for values that may be int/str/None.
    Ensures mixed types don't trigger TypeErrors during sorting.
    """
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value) if value is not None else '')


def build_schedule_payload(race_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds a lightweight payload describing available dates and tracks.

    Args:
        race_data: Parsed configuration dictionary

    Returns:
        Dict: JSON-friendly schedule payload
    """
    dates = []

    for date_entry in race_data.get('dates', []):
        tracks = []
        for track in date_entry.get('tracks', []):
            tracks.append({
                'id': track.get('id'),
                'name': track.get('name', track.get('id')),
                'location': track.get('location', ''),
                'race_count': len(track.get('races', []))
            })

        dates.append({
            'id': date_entry.get('id'),
            'label': date_entry.get('label', date_entry.get('id')),
            'tracks': tracks
        })

    return {'dates': dates}


def find_date_entry(race_data: Dict[str, Any], date_id: str) -> Optional[Dict[str, Any]]:
    """
    Returns the date entry matching the provided identifier.
    """
    for date_entry in race_data.get('dates', []):
        if date_entry.get('id') == date_id:
            return date_entry
    return None


def find_track_entry(date_entry: Dict[str, Any], track_id: str) -> Optional[Dict[str, Any]]:
    """
    Returns the track entry for a given date.
    """
    for track_entry in date_entry.get('tracks', []):
        if track_entry.get('id') == track_id:
            return track_entry
    return None


def build_horse_feature_row(
    horse: Dict[str, Any],
    race_meta: Dict[str, Any],
    date_id: str,
    track_id: str
) -> Dict[str, float]:
    """
    Builds a placeholder feature vector for a single horse.

    The features are deterministic based on date/track/horse so results are stable
    when mocking the model. Replace this logic with real feature engineering later.
    """
    seed = f"{date_id}-{track_id}-{race_meta.get('race_number')}-{horse.get('horse_id') or horse.get('horse_name') or horse.get('name')}"
    rng = random.Random(seed)

    post_position_value = horse.get('number')
    if post_position_value is None:
        post_position_value = horse.get('program_number')
    try:
        post_position_value = float(post_position_value)
    except (TypeError, ValueError):
        post_position_value = float(rng.randint(1, 12))

    return {
        'pace_early': round(rng.uniform(0.25, 1.0), 3),
        'pace_late': round(rng.uniform(0.25, 1.0), 3),
        'stamina_score': round(rng.uniform(0.25, 1.0), 3),
        'surface_fit': round(rng.uniform(0.25, 1.0), 3),
        'post_position': post_position_value,
        'class_rating': round(rng.uniform(0.25, 1.0), 3)
    }


def normalize_predictions(raw_predictions: Any, expected_length: int) -> List[float]:
    """
    Normalizes raw model outputs to a list of probabilities that sum to 1.

    Args:
        raw_predictions: Response payload from the model endpoint
        expected_length: Number of horses in the race

    Returns:
        List[float]: Normalized probabilities
    """
    predictions = raw_predictions

    if isinstance(predictions, dict):
        if 'predictions' in predictions:
            predictions = predictions['predictions']
        elif 'outputs' in predictions:
            predictions = predictions['outputs']

    if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
        predictions = [row[0] if isinstance(row, list) and row else row for row in predictions]

    if not isinstance(predictions, list):
        raise ValueError('Model response is not a list of predictions.')

    if len(predictions) != expected_length:
        raise ValueError(
            f"Model returned {len(predictions)} predictions, expected {expected_length}."
        )

    cleaned = [max(float(value), 0.0001) for value in predictions]
    total = sum(cleaned) or 1.0
    return [value / total for value in cleaned]


def score_model(dataset: pd.DataFrame) -> List[float]:
    """
    Sends a prediction request to the MLflow model serving endpoint.

    Args:
        dataset: pandas DataFrame containing the input features

    Returns:
        List[float]: The prediction result from the model

    Raises:
        Exception: If the API request fails or returns an error
    """
    url = Config.MLFLOW_ENDPOINT_URL
    token = Config.DATABRICKS_TOKEN
    timeout = Config.REQUEST_TIMEOUT

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    data_dict = {'dataframe_split': dataset.to_dict(orient='split')}
    data_json = json.dumps(data_dict, allow_nan=True)

    try:
        response = requests.post(
            url=url,
            headers=headers,
            data=data_json,
            timeout=timeout
        )
    except requests.exceptions.Timeout:
        raise Exception(f'Request timed out after {timeout} seconds.')
    except requests.exceptions.ConnectionError:
        raise Exception('Failed to connect to the model endpoint.')

    if response.status_code != 200:
        raise Exception(
            f'Request failed with status {response.status_code}. '
            f'Response: {response.text}'
        )

    return response.json()


COLOR_PALETTE = [
    "#0ea5e9",  # blue
    "#a855f7",  # purple
    "#10b981",  # emerald
    "#f97316",  # orange
    "#ef4444",  # red
    "#6366f1",  # indigo
    "#f59e0b",  # amber
    "#14b8a6",  # teal
    "#8b5cf6",  # violet
    "#22c55e",  # green
    "#e11d48",  # rose
    "#7c3aed",  # deep violet
    "#06b6d4",  # cyan
    "#ea580c",  # burnt orange
    "#facc15",  # yellow
    "#0f172a",  # navy
    "#84cc16",  # lime
    "#fb7185",  # pink
    "#1d4ed8",  # royal blue
    "#f472b6",  # hot pink
    "#2dd4bf",  # aqua
    "#c084fc",  # lavender
    "#34d399",  # mint
    "#fbbf24",  # gold
    "#f43f5e",  # raspberry
    "#22d3ee",  # sky
    "#3b82f6",  # azure
]


def assign_color(horse: Dict[str, Any]) -> str:
    """
    Returns an existing color or deterministically assigns one when missing.
    """
    if horse.get('color'):
        return horse['color']

    seed_value = horse.get('horse_id') or horse.get('horse_name') or horse.get('name') or horse.get('number') or random.random()
    rng = random.Random(str(seed_value))
    return rng.choice(COLOR_PALETTE)


def mock_probabilities(horses: List[Dict[str, Any]], seed_token: str) -> List[float]:
    """
    Generates deterministic mock probabilities for a list of horses.

    This is used when no model endpoint is configured or when requests fail.
    """
    rng = random.Random(seed_token)
    base_scores = [rng.uniform(0.3, 1.0) for _ in horses]
    total = sum(base_scores) or 1.0
    return [score / total for score in base_scores]


def generate_predictions(date_id: str, track_id: str, include_predictions: bool = True) -> Dict[str, Any]:
    """
    Generates race payload for a given date and track, optionally including model probabilities.
    """
    race_data = load_race_config()
    winners_data = load_winners_config()
    winners_by_race = {
        winner.get('race_id'): winner
        for winner in winners_data.get('winners', [])
    }
    date_entry = find_date_entry(race_data, date_id)

    if not date_entry:
        raise ValueError(f"Date '{date_id}' was not found in the schedule.")

    track_entry = find_track_entry(date_entry, track_id)
    if not track_entry:
        raise ValueError(f"Track '{track_id}' was not found for date {date_id}.")

    races_payload: List[Dict[str, Any]] = []
    races = sorted(
        track_entry.get('races', []),
        key=lambda race: (
            race.get('post_time') or '',
            sortable_number(race.get('race_number'))
        )
    )

    for race in races:
        horses = sorted(
            race.get('horses', []),
            key=lambda h: sortable_number(h.get('number'))
        )

        if not horses:
            races_payload.append({
                'race_id': f"{track_id}-{race.get('race_number')}",
                'race_number': race.get('race_number'),
                'post_time': race.get('post_time'),
                'distance': race.get('distance'),
                'surface': race.get('surface'),
                'purse': race.get('purse'),
                'class': race.get('class'),
                'winner': race.get('winner'),
                'field_size': 0,
                'model_top_pick_id': None,
                'horses': []
            })
            continue

        palette = COLOR_PALETTE.copy()
        rng_palette = random.Random(f"{date_id}-{track_id}-{race.get('race_number')}-palette")
        rng_palette.shuffle(palette)

        horses_enriched = []
        for idx, horse in enumerate(horses):
            chosen_color = horse.get('color') or palette[idx % len(palette)]
            post_position_value = horse.get('post_position')
            if post_position_value is None:
                post_position_value = horse.get('number')
            if post_position_value is None:
                post_position_value = horse.get('program_number')

            horses_enriched.append({
                'horse_id': horse.get('horse_id'),
                'horse_name': horse.get('horse_name') or horse.get('name'),
                'number': horse.get('number'),
                'post_position': post_position_value,
                'color': chosen_color,
                'probability': None
            })

        model_top_pick_id = None
        model_top_pick_name = None
        race_id = race.get('race_id') or f"{track_id}-{race.get('race_number')}"
        winner_entry = winners_by_race.get(race_id, {})
        winner_horse_id = winner_entry.get('winner_horse_id') or race.get('winner_horse_id') or race.get('winner')
        winner_name = winner_entry.get('winner_name') or race.get('winner_name') or race.get('winner')

        # If the config didn't include a winner name, try to derive it from the field
        if not winner_name and winner_horse_id:
            for horse in horses_enriched:
                if horse.get('horse_id') == winner_horse_id:
                    winner_name = horse.get('horse_name')
                    break

        if include_predictions:
            feature_rows = [
                build_horse_feature_row(horse, race, date_id, track_id)
                for horse in horses
            ]

            dataset = pd.DataFrame(feature_rows, columns=Config.MODEL_FEATURES)

            try:
                needs_mock = (
                    Config.USE_MOCK_MODEL
                    or not Config.MLFLOW_ENDPOINT_URL
                    or not Config.DATABRICKS_TOKEN
                )

                if needs_mock:
                    probabilities = mock_probabilities(
                        horses,
                        f"{date_id}-{track_id}-{race.get('race_number', 'race')}"
                    )
                else:
                    raw_predictions = score_model(dataset)
                    probabilities = normalize_predictions(raw_predictions, len(horses))

            except Exception as exc:  # noqa: BLE001 - bubble unexpected errors to the client
                print(f"[WARN] Falling back to mock predictions: {exc}")
                probabilities = mock_probabilities(
                    horses,
                    f"{date_id}-{track_id}-{race.get('race_number', 'race')}"
                )

            for horse_entry, prob in zip(horses_enriched, probabilities):
                horse_entry['probability'] = round(prob, 4)

            top_pick = max(horses_enriched, key=lambda h: h['probability'] or 0)
            model_top_pick_id = top_pick.get('horse_id')
            model_top_pick_name = top_pick.get('horse_name')

        races_payload.append({
            'race_id': f"{track_id}-{race.get('race_number')}",
            'race_number': race.get('race_number'),
            'post_time': race.get('post_time'),
            'distance': race.get('distance'),
            'surface': race.get('surface'),
            'purse': race.get('purse'),
            'class': race.get('class'),
            'winner_horse_id': winner_horse_id,
            'winner_name': winner_name,
            'field_size': len(horses),
            'model_top_pick_id': model_top_pick_id,
            'model_top_pick_name': model_top_pick_name,
            'horses': horses_enriched
        })

    return {
        'date': {
            'id': date_id,
            'label': date_entry.get('label', date_id)
        },
        'track': {
            'id': track_id,
            'name': track_entry.get('name', track_id),
            'location': track_entry.get('location')
        },
        'races': races_payload
    }


# ============================================================================
# Flask Routes
# ============================================================================


@app.route('/')
def home():
    """
    Serves the main application page.
    """
    return render_template('index.html')


@app.route('/schedule', methods=['GET'])
def schedule():
    """
    Returns the available dates and tracks defined in the text configuration file.
    """
    try:
        race_data = load_race_config()
        payload = build_schedule_payload(race_data)
        return jsonify({'success': True, **payload})
    except Exception as exc:  # noqa: BLE001 - send error to the client
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/card', methods=['POST'])
def card():
    """
    Returns race and horse information for a selected date and track without probabilities.
    """
    data = request.get_json(silent=True) or {}
    date_id = data.get('date_id')
    track_id = data.get('track_id')

    if not date_id or not track_id:
        return jsonify({
            'success': False,
            'error': 'Both date_id and track_id are required.'
        }), 400

    try:
        result = generate_predictions(date_id, track_id, include_predictions=False)
        return jsonify({'success': True, **result})
    except Exception as exc:  # noqa: BLE001 - return error to client
        return jsonify({
            'success': False,
            'error': str(exc)
        }), 400


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction requests for a selected date and track.

    Request Format:
        POST /predict
        Content-Type: application/json
        Body: { "date_id": "2024-07-04", "track_id": "belmont-park" }
    """
    data = request.get_json(silent=True) or {}
    date_id = data.get('date_id')
    track_id = data.get('track_id')

    if not date_id or not track_id:
        return jsonify({
            'success': False,
            'error': 'Both date_id and track_id are required.'
        }), 400

    try:
        result = generate_predictions(date_id, track_id)
        return jsonify({'success': True, **result})
    except Exception as exc:  # noqa: BLE001 - return error to client
        return jsonify({
            'success': False,
            'error': str(exc)
        }), 400


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring and deployment systems.
    """
    is_valid, errors = Config.validate_config()

    if is_valid:
        return jsonify({
            'status': 'healthy',
            'app': Config.APP_NAME,
            'version': Config.APP_VERSION
        }), 200

    return jsonify({
        'status': 'unhealthy',
        'errors': errors
    }), 500


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == '__main__':
    """
    Main entry point when running the application directly.
    """
    print("\n" + "="*70)
    print("STARTING RACE DAY PREDICTION APPLICATION")
    print("="*70)

    # Validate configuration before starting the server
    is_valid, errors = Config.validate_config()

    if not is_valid:
        print("\n❌ CONFIGURATION ERROR!\n")
        print("The application cannot start due to configuration issues:\n")
        for error in errors:
            print(f"  • {error}")
        print("\nPlease check your .env file or environment variables.")
        sys.exit(1)

    print("\n✅ Configuration validated successfully!\n")
    print(f"🚀 Starting server at http://{Config.HOST}:{Config.PORT}")
    print(f"📊 Model endpoint: {Config.MLFLOW_ENDPOINT_URL}")
    print("\nPress CTRL+C to stop the server\n")
    print("="*70 + "\n")

    app.run(
        debug=Config.DEBUG,
        host=Config.HOST,
        port=Config.PORT
    )
