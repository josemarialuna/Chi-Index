from datetime import datetime
import time


def whatTimeIsIt(time_format='d/m/Y'):
    """Returns the timestamp in the time_format given."""
    return datetime.datetime.fromtimestamp(time.time()).strftime(time_format)


def save_dataframe(df, results_path: str, filename: str) -> None:
    """Save the dataframe (df) into the self.results_path with the 'filename'."""
    df.to_csv(f'{results_path}/{filename}', sep='\t')
