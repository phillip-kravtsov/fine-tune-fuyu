import subprocess
import os
import argparse

OUTPUT_DIR = '/workspace/fine-tune-fuyu/output'
BUCKET_NAME = 'philkrav-bucket'

def export_experiment(run_name, output_dir, bucket_name):
    run_dir = f'{output_dir}/{run_name}'
    assert os.path.exists(run_dir), f'Run dir {run_dir} does not exist'
    bucket_path = f's3://{bucket_name}/fuyu/output/'
    subprocess.run(['s5cmd', 'cp', run_dir, bucket_path])

def import_experiment(run_name, output_dir, bucket_name, step):
    run_dir = f'{output_dir}/{run_name}'
    if os.path.exists(run_dir):
        print(f'Run dir {run_dir} already exists')
        return
    if step is None:
        bucket_path = f's3://{bucket_name}/fuyu/output/{run_name}/*'
        subprocess.run(['s5cmd', 'cp', bucket_path, run_dir])
    else:
        bucket_path = f's3://{bucket_name}/fuyu/output/{run_name}/step-{step}/*'
        subprocess.run(['s5cmd', 'cp', bucket_path, os.path.join(run_dir, f'step-{step}')])

        bucket_path = f's3://{bucket_name}/fuyu/output/{run_name}/config.json'
        subprocess.run(['s5cmd', 'cp', bucket_path, run_dir])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['push', 'pull'])
    parser.add_argument('run_name')
    parser.add_argument('--bucket', default=BUCKET_NAME)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--step', default=None)
    args = parser.parse_args()
    if args.command == 'push':
        export_experiment(args.run_name, args.output_dir, args.bucket, args.step)
    elif args.command == 'pull':
        import_experiment(args.run_name, args.output_dir, args.bucket, args.step)
