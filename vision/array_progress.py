import argparse
import signal
import submitit
import time
import sys

def main(args):
    """
    Entry point of the script, retrieves job information and tracks progress of a Submitit job array.

    Args:
        args (argparse.Namespace): The command-line arguments.

    This function retrieves the jobs associated with the provided job array ID and tracks their progress.
    It displays a progress bar and job status counts, updating at the specified interval until all jobs are finished.
    """

    # Retrieve the jobs using the provided array ID
    jobs = submitit.helpers.query_jobs(job_id=args.array_id)

    if not jobs:
        print(f"No jobs found with array ID: {args.array_id}")
        return

    # Initialize progress variables
    num_jobs = len(jobs)
    jobs_ended = 0
    status_msg_len = 0

    # Define handler for SIGINT (Ctrl-C) to exit gracefully
    def signal_handler(sig, frame):
        print("\nExiting...")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # Loop until all jobs are finished
    while jobs_ended < num_jobs:
        job_stats = {
            'PENDING': 0,
            'RUNNING': 0, 
            'COMPLETED': 0,
            'FAILED': 0,
            'CANCELLED': 0,
            'TIMEOUT': 0,
            'NODE_FAIL': 0,
            'UNKNOWN': 0
        }

        # Update job status counts
        for job in jobs:
            job_stats[job.state] += 1
        jobs_ended = job_stats['COMPLETED'] + job_stats['CANCELLED'] + job_stats['FAILED'] + job_stats['TIMEOUT']
        
        # Update progress bar
        progress_bar = "[{0}{1}] {2}/{3} jobs finished".format(
            "=" * (jobs_ended * 30 // num_jobs),
            " " * ((num_jobs - jobs_ended) * 30 // num_jobs),
            jobs_ended,
            num_jobs
        )

        # Format job status counts into a printable string
        state_count_labels = {
            'COMPLETED':'Completed', 
            'RUNNING':'Running', 
            'PENDING':'Pending', 
            'FAILED':'Failed',
            'CANCELLED':'Cancelled', 
            'TIMEOUT':'Timeout', 
            'NODE_FAIL':'Node Failure', 
            'UNKNOWN':'Unknown'
        }
        state_count_nonzero = [f"{state_count_labels[state]}: {job_stats[state]}" 
                               for state in job_stats if job_stats[state] > 0]

        # Print progress bar and status counts
        status_msg = f"\r{progress_bar} ({', '.join(state_count_nonzero)})"
        print(status_msg + " "*max(status_msg_len-len(status_msg), 0), end="", flush=True)
        status_msg_len = len(status_msg)

        # Wait before next update
        time.sleep(args.interval)

    print("\nJob array completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Track progress of a submitit job array.')
    parser.add_argument('array_id', help='Job array ID')
    parser.add_argument('--interval', type=int, default=1, help='Status update interval in seconds (default: 1)')
    args = parser.parse_args()
    main(args)