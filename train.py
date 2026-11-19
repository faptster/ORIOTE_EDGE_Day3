"""Reproduce the saved training configuration with an externally exported dataset."""
from pathlib import Path
import argparse,json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--epochs',type=int)
    p.add_argument('--batch',type=int)
    p.add_argument('--device')
    args=p.parse_args()
    config=json.loads((Path(__file__).parent/'training_config.json').read_text())
    kwargs=dict(config['kwargs'])
    for key in ('epochs','batch','device'):
        value=getattr(args,key)
        if value is not None:kwargs[key]=value
    if not Path(args.data).is_file():raise FileNotFoundError(args.data)
    data_path=Path(args.data).resolve()
    if (data_path.parent/'images/train').is_dir() and (data_path.parent/'images/val').is_dir():
        # The exported dataset may have moved from /content to another machine.
        import yaml
        data=yaml.safe_load(data_path.read_text());data['path']=str(data_path.parent)
        data_path=data_path.parent/'data_local.yaml'
        data_path.write_text(yaml.safe_dump(data,sort_keys=False))
    kwargs.update(data=str(data_path),project='runs',name='beans_reproduce',exist_ok=False)
    from ultralytics import YOLO
    YOLO(config['model']).train(**kwargs)

if __name__=='__main__':main()
