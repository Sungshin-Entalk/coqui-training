import os
import sys
import torch
import io

# 필요한 클래스들을 정확한 경로에서 import
from trainer.trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# stdout의 인코딩을 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def formatter(root_path, manifest_file, **kwargs):
    """Assumes each line as ```<filename>.wav|<transcription>```"""
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

def main():
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
        test_sentences=[]
    )

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

    # 트레이너 초기화
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    # 학습 시작
    trainer.fit()

if __name__ == '__main__':
    main()