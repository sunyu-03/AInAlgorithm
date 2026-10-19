"""Re-score unchanged current-layer routes at the committed reservation state.

Wait includes all W time since own previous completion, as in the old scorer;
it is not claimed to isolate collision-only delay. No route geometry is changed.
"""
def current_scores(state, reference, sequences, z):
    waits, finishes = {}, {}
    for robot, rr in reference['model'].rows.items():
        offset = reference['offsets'][robot]
        wait = offset - state.ends[robot]
        if wait < -1e-6:
            raise ValueError('Refreshed route starts before robot is available')
        wait = max(0., wait)
        ids = [item[0] for item in sequences[robot]]
        index = 0
        for a, b in zip(rr, rr[1:]):
            if a[4] == 'W':
                wait += b[0] - a[0]
            elif a[4] == 'D':
                waits[ids[index]] = wait
                finishes[z, robot] = offset + b[0]
                index += 1
                wait = 0.
        if index != len(ids):
            raise ValueError('Current layer task count mismatch')
    return waits, finishes
