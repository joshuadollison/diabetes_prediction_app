/**
 * Race Day Predictor - Frontend
 *
 * Handles schedule loading, prediction requests, race rendering, and simulations.
 */

document.addEventListener('DOMContentLoaded', () => {
    bootstrapApp();
});

const appState = {
    schedule: null,
    selectedDate: null,
    selectedTrack: null,
    races: [],
    hasPredictions: false,
    modelChoice: 'F1'
};

async function bootstrapApp() {
    bindControls();
    await loadSchedule();
}

function bindControls() {
    const dateSelect = document.getElementById('dateSelect');
    const trackSelect = document.getElementById('trackSelect');
    const predictButton = document.getElementById('predictButton');
    const playAllButton = document.getElementById('playAllButton');
    const modelSelect = document.getElementById('modelSelect');

    if (dateSelect) {
        dateSelect.addEventListener('change', handleDateChange);
    }
    if (trackSelect) {
        trackSelect.addEventListener('change', handleTrackChange);
    }
    if (predictButton) {
        predictButton.addEventListener('click', handlePredictClick);
    }
    if (playAllButton) {
        playAllButton.addEventListener('click', handlePlayAllClick);
    }
    if (modelSelect) {
        modelSelect.addEventListener('change', handleModelChange);
    }
}

async function loadSchedule() {
    try {
        setButtonLoading(true);
        const response = await fetch('/schedule');
        const payload = await response.json();

        if (!response.ok || !payload.success) {
            throw new Error(payload.error || 'Failed to load schedule.');
        }

        appState.schedule = payload;
        populateDateOptions(payload.dates || []);
    } catch (error) {
        showAlert(error.message || 'Unable to load schedule.', 'error');
    } finally {
        setButtonLoading(false);
    }
}

function populateDateOptions(dates) {
    const dateSelect = document.getElementById('dateSelect');
    if (!dateSelect) return;

    if (!dates.length) {
        dateSelect.innerHTML = '<option value="">No dates configured</option>';
        return;
    }

    dateSelect.innerHTML = dates
        .map(date => `<option value="${date.id}">${date.label}</option>`)
        .join('');

    appState.selectedDate = dates[0]?.id || null;
    dateSelect.value = appState.selectedDate;

    populateTrackOptions(dates[0]?.tracks || []);
    if (appState.selectedDate && appState.selectedTrack) {
        loadCard(appState.selectedDate, appState.selectedTrack);
    }
}

function populateTrackOptions(tracks) {
    const trackSelect = document.getElementById('trackSelect');
    if (!trackSelect) return;

    if (!tracks.length) {
        trackSelect.innerHTML = '<option value="">No tracks for this date</option>';
        appState.selectedTrack = null;
        return;
    }

    trackSelect.innerHTML = tracks
        .map(track => `<option value="${track.id}">${track.name} (${track.race_count} races)</option>`)
        .join('');

    appState.selectedTrack = tracks[0]?.id || null;
    trackSelect.value = appState.selectedTrack;
}

function handleDateChange(event) {
    const selectedDate = event.target.value;
    appState.selectedDate = selectedDate || null;
    appState.selectedTrack = null;
    appState.races = [];
    appState.hasPredictions = false;
    renderRaceGrid([], null, false);

    const dateEntry = appState.schedule?.dates?.find(date => date.id === selectedDate);
    populateTrackOptions(dateEntry?.tracks || []);

    if (appState.selectedDate && appState.selectedTrack) {
        loadCard(appState.selectedDate, appState.selectedTrack);
    }
}

function handleTrackChange(event) {
    appState.selectedTrack = event.target.value || null;
    appState.hasPredictions = false;
    if (appState.selectedDate && appState.selectedTrack) {
        loadCard(appState.selectedDate, appState.selectedTrack);
    }
}

async function handlePredictClick() {
    hideAlert();
    const { selectedDate, selectedTrack } = appState;

    if (!selectedDate || !selectedTrack) {
        showAlert('Choose a date and track to fetch predictions.', 'warning');
        return;
    }

    try {
        setButtonLoading(true);
        const payload = await fetchPredictions(selectedDate, selectedTrack, appState.modelChoice);
        appState.races = payload.races || [];
        appState.hasPredictions = true;
        renderRaceGrid(appState.races, payload, true);
    } catch (error) {
        showAlert(error.message || 'Prediction request failed.', 'error');
    } finally {
        setButtonLoading(false);
    }
}

function handlePlayAllClick() {
    hideAlert();
    if (!appState.races || appState.races.length === 0) {
        showAlert('Load a card first.', 'warning');
        return;
    }
    appState.races.forEach(race => {
        if (race && race.race_id) {
            playRaceAnimation(race.race_id);
        }
    });
}

function handleModelChange(event) {
    appState.modelChoice = event.target.value || 'F1';
}

async function loadCard(dateId, trackId) {
    try {
        setButtonLoading(true);
        const response = await fetch('/card', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ date_id: dateId, track_id: trackId })
        });

        const payload = await response.json();
        if (!response.ok || !payload.success) {
            throw new Error(payload.error || 'Failed to load card data.');
        }

        appState.races = payload.races || [];
        appState.hasPredictions = false;
        renderRaceGrid(appState.races, payload, false);
    } catch (error) {
        showAlert(error.message || 'Unable to load card data.', 'error');
    } finally {
        setButtonLoading(false);
    }
}

async function fetchPredictions(dateId, trackId) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date_id: dateId, track_id: trackId, model_choice: appState.modelChoice })
    });

    const payload = await response.json();

    if (!response.ok || !payload.success) {
        throw new Error(payload.error || 'Prediction request failed.');
    }

    return payload;
}

function renderRaceGrid(races, payloadMeta, hasPredictions) {
    const grid = document.getElementById('raceGrid');
    const emptyState = document.getElementById('emptyState');

    if (!grid) return;

    if (!races.length) {
        grid.innerHTML = '';
        if (emptyState) emptyState.classList.remove('hidden');
        return;
    }

    if (emptyState) emptyState.classList.add('hidden');

    grid.innerHTML = races.map(race => buildRaceCard(race, payloadMeta, hasPredictions)).join('');
    attachRaceHandlers();
}

function buildRaceCard(race, payloadMeta, hasPredictions) {
    const horsesWithPositions = (race.horses || []).filter(horse => horse.post_position != null);
    const horses = sortHorsesForDisplay(horsesWithPositions, hasPredictions);
    const fieldSize = typeof race.field_size === 'number' ? race.field_size : horses.length;
    const metaLine = [
        race.surface || 'Surface TBD',
        race.distance || 'Distance TBD',
        race.purse || 'Purse TBD'
    ].join(' • ');

    const headerLine = `${payloadMeta?.track?.name || 'Track'} • ${payloadMeta?.date?.label || ''}`;

    const horseList = horses.map((horse, index) => {
        let probabilityText = 'Probability pending';
        if (horse.probability != null && horse.probability_raw != null) {
            probabilityText = `${(horse.probability * 100).toFixed(1)}% (normalized) • ${(horse.probability_raw * 100).toFixed(2)}% raw`;
        } else if (horse.probability != null) {
            probabilityText = `${(horse.probability * 100).toFixed(1)}% win chance`;
        }
        const isTopPick = hasPredictions && index === 0;
        const highlightClass = isTopPick ? 'horse-pick' : 'bg-white/5';
        const numberContent = horse.post_position != null ? `${horse.post_position}` : '';

        return `
        <li class="horse-row flex items-start gap-3 px-3 py-2 rounded-lg ${highlightClass}" data-horse-id="${horse.horse_id || ''}" data-top-pick="${isTopPick}">
            <div class="horse-chip" style="background:${horse.color || '#334155'}">
                ${numberContent ? `<span class="horse-chip__number">${numberContent}</span>` : ''}
            </div>
            <div class="flex-1 space-y-1">
                <div class="horse-label-stack">
                    ${isTopPick ? '<span class="pill pill--pick">Model pick</span>' : ''}
                </div>
                <div class="font-semibold text-white">${horse.horse_name || horse.name}</div>
                <div class="text-xs text-slate-300">${probabilityText}</div>
            </div>
        </li>
    `;
    }).join('');

    const lanes = horses.map((horse, idx) => `
        <div class="horse-lane" data-index="${idx}">
            <div class="horse-icon" style="background:${horse.color || '#334155'}"
                 data-horse-id="${horse.horse_id}" data-post-position="${horse.post_position ?? ''}" data-probability="${horse.probability ?? ''}">
                ${horse.post_position != null ? `<span class="horse-number">${horse.post_position}</span>` : ''}
            </div>
        </div>
    `).join('');

    const modelPickLabel = hasPredictions && race.model_top_pick_name
        ? `Model top pick: ${race.model_top_pick_name}`
        : 'Run predictions to pick a favorite';

    return `
        <article class="race-card" data-race-id="${race.race_id}" data-winner="${race.winner_horse_id || ''}" data-winner-name="${race.winner_name || ''}" data-model-pick-id="${race.model_top_pick_id || ''}">
            <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div>
                    <p class="text-xs uppercase tracking-[0.15em] text-brand-200">${headerLine}</p>
                    <h2 class="text-2xl font-display font-semibold text-white">Race ${race.race_number} • ${race.post_time || 'TBD'}</h2>
                    <p class="text-sm text-slate-300">${metaLine}</p>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-5 mt-5">
                <div class="space-y-3">
                    <div class="flex items-center justify-between">
                        <h3 class="text-sm font-semibold text-slate-200">${hasPredictions ? 'Ranked by probability' : 'Field (config order)'}</h3>
                        <span class="text-xs text-slate-400">Field size: ${fieldSize}</span>
                    </div>
                    <ul class="space-y-2">
                        ${horseList}
                    </ul>
                </div>

                <div class="space-y-3">
                    <div class="flex items-center justify-between">
                        <div class="text-sm font-semibold text-slate-200">Simulation</div>
                        <div class="pill pill--ghost text-xs">${modelPickLabel}</div>
                    </div>
                    <div class="race-track" data-race-id="${race.race_id}">
                        <div class="finish-line" aria-hidden="true"></div>
                        ${lanes}
                        <div class="race-outcome" aria-live="polite">
                            <span class="race-outcome__text"></span>
                        </div>
                    </div>
                    <div class="flex items-center justify-between gap-3">
                        <div class="text-xs text-slate-300">
                            Play to reveal the configured winner after the run.
                        </div>
                        <button class="play-race-btn pill pill--brand" data-race-id="${race.race_id}" type="button">Play simulation</button>
                    </div>
                </div>
            </div>
        </article>
    `;
}

function sortHorsesForDisplay(horses, hasPredictions) {
    if (!hasPredictions) {
        return [...horses]
            .filter(horse => horse.post_position != null)
            .sort((a, b) => (a.post_position || 0) - (b.post_position || 0));
    }
    return [...horses]
        .filter(horse => horse.post_position != null)
        .sort((a, b) => (b.probability || 0) - (a.probability || 0));
}

function attachRaceHandlers() {
    const buttons = document.querySelectorAll('.play-race-btn');
    buttons.forEach(button => {
        button.addEventListener('click', () => {
            const raceId = button.getAttribute('data-race-id');
            playRaceAnimation(raceId);
        });
    });
}

function playRaceAnimation(raceId) {
    const race = appState.races.find(entry => entry.race_id === raceId);
    const track = document.querySelector(`.race-track[data-race-id="${raceId}"]`);
    const card = document.querySelector(`.race-card[data-race-id="${raceId}"]`);

    if (!race || !track || !card) return;

    const winnerId = card.getAttribute('data-winner');
    const winnerName = card.getAttribute('data-winner-name');
    const modelPickId = card.getAttribute('data-model-pick-id');
    const horses = track.querySelectorAll('.horse-icon');

    track.classList.remove('running', 'result-win', 'result-lose', 'show-result');
    clearWinnerMarks(card);

    horses.forEach((horse, index) => {
        horse.classList.remove('is-winner', 'is-runner');
        horse.style.animation = 'none';
        void horse.offsetWidth;
        horse.style.animation = '';

        const isWinner = horse.dataset.horseId === winnerId;
        horse.classList.add(isWinner ? 'is-winner' : 'is-runner');
        horse.style.setProperty('--run-duration', isWinner ? '4.2s' : `${4.8 + index * 0.12}s`);
        horse.style.setProperty('--lane-index', index);
    });

    const outcomeText = track.querySelector('.race-outcome__text');
    if (outcomeText) outcomeText.textContent = '';

    track.classList.add('running');

    window.setTimeout(() => {
        const hasPredictions = appState.hasPredictions;
        const modelHit = hasPredictions && winnerId && modelPickId && winnerId === modelPickId;

        track.classList.add('show-result');
        if (hasPredictions) {
            track.classList.toggle('result-win', modelHit);
            track.classList.toggle('result-lose', !modelHit);
            if (outcomeText) {
                outcomeText.textContent = modelHit ? 'WINNER' : 'LOSER';
            }
        } else {
            track.classList.remove('result-win', 'result-lose');
        }

        revealWinner(card, winnerId, modelPickId, winnerName);
    }, 4400);
}

function clearWinnerMarks(card) {
    const rows = card.querySelectorAll('.horse-row');
    rows.forEach(row => {
        row.classList.remove('horse-winner', 'horse-pick-lose');
        const labelStack = row.querySelector('.horse-label-stack');
        if (labelStack) {
            labelStack.querySelectorAll('.crown-icon').forEach(el => el.remove());
            labelStack.querySelectorAll('.pill').forEach(el => el.remove());
            if (row.dataset.topPick === 'true') {
                const pickPill = document.createElement('span');
                pickPill.className = 'pill pill--pick';
                pickPill.textContent = 'Model pick';
                labelStack.appendChild(pickPill);
            }
        }
    });
}

function revealWinner(card, winnerId, modelPickId, winnerName) {
    if (!winnerId) return;
    const rows = card.querySelectorAll('.horse-row');
    rows.forEach(row => {
        const rowId = row.getAttribute('data-horse-id');
        const labelStack = row.querySelector('.horse-label-stack');
        if (!labelStack) return;

        if (rowId === winnerId) {
            row.classList.add('horse-winner');
            row.classList.remove('horse-pick');

            labelStack.innerHTML = '';
            const winnerPill = document.createElement('span');
            winnerPill.className = 'pill pill--winner';
            winnerPill.textContent = 'Winner';
            labelStack.appendChild(winnerPill);

            if (rowId === modelPickId && !labelStack.querySelector('.crown-icon')) {
                const crown = document.createElement('span');
                crown.className = 'crown-icon';
                crown.textContent = '👑';
                labelStack.prepend(crown);
            }
        } else if (rowId === modelPickId) {
            row.classList.add('horse-pick-lose');
            labelStack.innerHTML = '<span class="pill pill--pick-loss">Model pick</span>';
        }
    });
}

function setButtonLoading(isLoading) {
    const button = document.getElementById('predictButton');
    if (!button) return;

    const label = button.querySelector('.button-label');
    const spinner = button.querySelector('.button-spinner');

    if (isLoading) {
        button.disabled = true;
        label?.classList.add('opacity-60');
        if (spinner) spinner.classList.remove('hidden');
    } else {
        button.disabled = false;
        label?.classList.remove('opacity-60');
        if (spinner) spinner.classList.add('hidden');
    }
}

function showAlert(message, type = 'info') {
    const alert = document.getElementById('alert');
    if (!alert) return;

    const styles = {
        info: 'bg-white/10 border-white/30 text-slate-100',
        warning: 'bg-amber-500/20 border-amber-400/50 text-amber-100',
        error: 'bg-red-500/20 border-red-400/50 text-red-100'
    };

    alert.className = `rounded-xl px-4 py-3 text-sm border ${styles[type] || styles.info}`;
    alert.textContent = message;
    alert.classList.remove('hidden');
}

function hideAlert() {
    const alert = document.getElementById('alert');
    if (alert) alert.classList.add('hidden');
}
