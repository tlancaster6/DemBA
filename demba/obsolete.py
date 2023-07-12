
def threshed_pelt_event_detection(data: pd.Series, upper_thresh: float, min_size: int, penalty=3):
    binary_data = (data < upper_thresh).astype(int).values
    bkpts = rpt.Pelt(model='l2', min_size=min_size, jump=1).fit_predict(binary_data, penalty)
    if bkpts[0] != 0:
        bkpts = [0] + bkpts
    if bkpts[-1] != len(data):
        bkpts = bkpts + [len(data)]
    event_ids = pd.Series(-1, index=data.index)
    current_event = 0
    for start, stop in list(zip(bkpts[:-1], bkpts[1:])):
        if np.mean(binary_data[start:stop]) > 0.5:
            event_ids.iloc[start:stop] = current_event
            current_event += 1
    return event_ids