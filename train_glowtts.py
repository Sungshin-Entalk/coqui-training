import os
import sys
import io
import torch
import torchaudio
import wandb  # wandb 임포트
import torch.nn.functional as F
import torch.optim as optim

from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# stdout의 인코딩을 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class TrainerArgs:
    pass

class Trainer:
    def __init__(self, args, config, output_path, model, train_samples, eval_samples):
        self.args = args
        self.config = config
        self.output_path = output_path
        self.model = model
        self.train_samples = train_samples
        self.eval_samples = eval_samples
        self.optimizer = None  # Placeholder for the optimizer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def load_audio(self, audio_file):
        waveform, sample_rate = torchaudio.load(audio_file)
        if sample_rate != self.config.audio.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.config.audio.sample_rate)(waveform)
        return waveform

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for sample in self.train_samples:
            self.optimizer.zero_grad()
            input_data = self.load_audio(sample['audio_file'])
            x_lengths = torch.tensor([input_data.shape[1]])
            target_data = self.load_audio(sample['audio_file'])  
            y = target_data.unsqueeze(0)
            output = self.model(input_data, x_lengths, y)
            loss = self.compute_loss(output, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(self.train_samples)
        print(f"Training Loss: {average_loss}")
        return average_loss

    def evaluate(self, samples):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for sample in samples:
                input_data = self.load_audio(sample['audio_file'])
                x_lengths = torch.tensor([input_data.shape[1]])
                target_data = self.load_audio(sample['audio_file'])  
                y = target_data.unsqueeze(0)
                output = self.model(input_data, x_lengths, y)
                loss = self.compute_loss(output, y)
                total_loss += loss.item()
        average_loss = total_loss / len(samples)
        return average_loss

    def compute_loss(self, output, target):
        loss = F.mse_loss(output, target)
        return loss

def formatter(root_path, manifest_file, **kwargs):
    """Assumes each line as 
<filename>.wav|<transcription>
"""
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    speaker_name = "my_Sherlock"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.strip().split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0])
            text = cols[1]
            # 디버깅을 위해 파일 경로와 텍스트 출력
            print(f"Processing file: {wav_file}, Text: {text}")
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items

def load_model(config, checkpoint_path):
    # 오디오 프로세서 초기화
    ap = AudioProcessor.init_from_config(config)
    
    # 토크나이저 초기화 및 설정 갱신
    tokenizer, config = TTSTokenizer.init_from_config(config)
    
    # GlowTTS 모델 초기화
    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)
    model.load_checkpoint(checkpoint_path)
    
    return model, ap, tokenizer

def infer(model, text, ap, tokenizer):
    # 입력 텍스트를 토큰으로 변환
    tokens = tokenizer.text_to_ids(text)
    
    # 토큰을 텐서로 변환하고 배치 차원 추가
    tokens = torch.LongTensor(tokens).unsqueeze(0)
    
    # 추론 모드 설정 (기울기 계산 비활성화)
    with torch.no_grad():
        # 모델 추론 수행
        outputs = model.inference(tokens)
    
    # 모델 출력을 이용해 오디오 파형 생성
    waveform = ap.invert_spectrogram(outputs["model_outputs"].squeeze(0))
    
    # 생성된 오디오 파형 반환
    return waveform

def perform_inference(config, checkpoint_path):
    # 모델, 오디오 프로세서, 토크나이저 초기화
    model, ap, tokenizer = load_model(config, checkpoint_path)
    
    # 사용자로부터 텍스트 입력 받기
    text = input("결과 text 내용을 적어주세요!: ")
    
    # 추론 수행
    waveform = infer(model, text, ap, tokenizer)
    
    # 결과 오디오 파일 저장
    output_file = os.path.join(config.output_path, "output_0807_1511.wav")  # 파일 형식: output_날짜_시간.wav 로 할 것!
    ap.save_wav(waveform, output_file)
    print(f"학습 이후 님이 설정한 text로 추론 오디오가 생성됐습니다~!..추론 오디오: {output_file}")

def main():
    # wandb 설정 및 초기화
    wandb.init(project="sherlock")
    wandb.require("core")  # 새로운 백엔드 사용 설정

    # torch의 기본 텐서 유형을 CPU 텐서로 설정
    torch.set_default_tensor_type(torch.FloatTensor)

    # 현재 스크립트의 디렉토리를 출력 경로로 설정
    output_path = os.path.dirname(os.path.abspath(__file__))

    # 데이터셋 설정 정의
    dataset_config = BaseDatasetConfig(
        formatter="Sherlock", 
        meta_file_train="metadata.txt",  # 올바른 경로로 수정
        path="MyTTSDataset"  # 올바른 경로로 수정
    )

    audio_config = BaseAudioConfig(
        sample_rate=12050,
    )

    character_config = CharactersConfig(
        characters_class="TTS.tts.utils.text.characters.Graphemes",
        pad="_",
        eos="~",
        bos="^",
        blank="@",
        characters="Iabdfgijklmnprstuvxzɔɛɣɨɫɱʂʐʲˈː̯͡β",
        punctuations="!,.?: -‒–—…",
    )

    # phoneme_cache_path 설정
    phoneme_cache_path = os.path.join(output_path, "phoneme_cache")

    # phoneme_cache 디렉토리 생성
    if not os.path.exists(phoneme_cache_path):
        os.makedirs(phoneme_cache_path)
        
    # 학습 설정 초기화
    config = GlowTTSConfig(
        run_name="Testrun",
        run_description="Desc",
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        audio=audio_config,
        characters=character_config,
        eval_split_size=0.2,
        test_sentences=[],
    )

    learning_rate=0.001

    # 오디오 프로세서 초기화
    ap = AudioProcessor.init_from_config(config)

    # 토크나이저 초기화
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # 데이터 샘플 로드
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        formatter=formatter,
        eval_split_size=config.eval_split_size,
        eval_split_max_size=config.eval_split_max_size,
    )

    # 모델 초기화
    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

    # 옵티마이저 초기화 (Adam 옵티마이저)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 트레이너 초기화
    trainer_args = TrainerArgs()
    trainer = Trainer(
    trainer_args, config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

    # 옵티마이저 설정
    trainer.set_optimizer(optimizer)

    # 학습 시작
    for epoch in range(config.epochs):
        train_loss = trainer.train_epoch()
        eval_loss = trainer.evaluate(eval_samples)
    
        # 학습 손실과 평가 손실 로그 기록
        wandb.log({"epoch": epoch, "train_loss": train_loss, "eval_loss": eval_loss})
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Eval Loss: {eval_loss}")
    

    # 모델 가중치 저장
    checkpoint_path = os.path.join(output_path, "checkpoints", "셜록_model.pth")  # 실제 체크포인트 경로로 수정
    torch.save(model.state_dict(), checkpoint_path)
    wandb.save(checkpoint_path)  # wandb에 모델 가중치 저장

    # 추론 수행
    perform_inference(config, checkpoint_path)

if __name__ == '__main__':
    main()