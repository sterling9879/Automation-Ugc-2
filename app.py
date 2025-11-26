"""
Aplicação Gradio - Interface do SaaS de Geração de Vídeos com Lip-Sync
Com suporte a múltiplos roteiros, preview, seleção de voz individual e imagem por batch
"""
import gradio as gr
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json

from config import Config
from job_manager import JobManager
from audio_generator import AudioGenerator
from utils import get_logger, split_into_paragraphs, create_batches

logger = get_logger(__name__)

# Estado global para armazenar dados do preview
preview_state = {
    "scripts": [],
    "batches_per_script": [],
    "images": [],
    "image_names": []
}

def get_audio_provider_choices() -> List[tuple]:
    """Retorna lista de provedores de áudio disponíveis"""
    providers = []

    if Config.ELEVENLABS_API_KEY:
        providers.append(("ElevenLabs (Text-to-Speech v3)", "elevenlabs"))

    if Config.MINIMAX_API_KEY:
        providers.append(("MiniMax Audio (Text-to-Speech)", "minimax"))

    if not providers:
        providers.append(("Nenhum provedor configurado", "none"))

    return providers

def get_voice_choices(provider: str = None) -> List[str]:
    """
    Obtém lista de vozes disponíveis do provedor especificado

    Args:
        provider: 'elevenlabs' ou 'minimax' (padrão: config)
    """
    try:
        if provider is None:
            provider = Config.AUDIO_PROVIDER

        if provider == 'none':
            return ["Configure uma API Key no arquivo .env"]

        audio_gen = AudioGenerator(provider=provider)
        voices = audio_gen.get_available_voices()

        if voices and len(voices) > 0:
            return [voice['name'] for voice in voices]
        else:
            logger.warning(f"Nenhuma voz disponível do {provider}")
            return [f"Configure a API Key do {provider} no arquivo .env"]

    except Exception as e:
        logger.error(f"Erro ao obter vozes do {provider}: {e}")
        print(f"\n  AVISO: Não foi possível conectar ao {provider}")
        print(f"   Verifique se a API Key no arquivo .env está correta")
        print(f"   Erro: {e}\n")
        return [f"Erro ao conectar - Verifique a API Key do {provider}"]

def update_voices_by_provider(provider: str):
    """
    Atualiza lista de vozes quando o provedor muda

    Args:
        provider: 'elevenlabs' ou 'minimax'

    Returns:
        gr.Dropdown.update com novas escolhas
    """
    voices = get_voice_choices(provider)
    return gr.Dropdown(choices=voices, value=voices[0] if voices else None)

def get_model_choices() -> List[tuple]:
    """Obtém lista de modelos ElevenLabs disponíveis"""
    return [
        ("Multilingual v3 (Mais recente, melhor qualidade)", "eleven_multilingual_v3"),
        ("Turbo v3 (Mais rápido, geração em tempo real)", "eleven_turbo_v3"),
        ("Flash v3 (Ultra rápido, baixa latência)", "eleven_flash_v3"),
        ("Multilingual v2 (Melhor qualidade v2)", "eleven_multilingual_v2"),
        ("Turbo v2.5 (Rápido e eficiente)", "eleven_turbo_v2_5"),
        ("Turbo v2 (Versão anterior rápida)", "eleven_turbo_v2"),
        ("Multilingual v1 (Legado)", "eleven_multilingual_v1"),
        ("Monolingual v1 (Inglês apenas)", "eleven_monolingual_v1"),
    ]

def estimate_job(text: str) -> str:
    """
    Estima custo e tempo do processamento

    Args:
        text: Texto de entrada

    Returns:
        String formatada com estimativas
    """
    try:
        if not text or not text.strip():
            return "Digite um texto para ver as estimativas"

        # Cria instância temporária para estimativa (provedor não importa para estimativa)
        temp_mgr = JobManager()
        estimate = temp_mgr.get_job_estimate(text)

        output = f"""
**Estimativa de Processamento**

**Análise do Texto:**
- Caracteres: {estimate['num_chars']:,}
- Batches: {estimate['num_batches']}
- Vídeos a gerar: {estimate['num_videos']}

**Tempo Estimado:** {estimate['estimated_time']}

**Custo Estimado:**
- Gemini (formatação): {estimate['estimated_cost']['gemini']}
- ElevenLabs (áudio): {estimate['estimated_cost']['elevenlabs']}
- WaveSpeed (vídeo): {estimate['estimated_cost']['wavespeed']}
- **Total: {estimate['estimated_cost']['total']}**

Os valores são aproximados e podem variar conforme uso real das APIs.
"""
        return output

    except Exception as e:
        logger.error(f"Erro ao estimar job: {e}")
        return f"Erro ao calcular estimativa: {str(e)}"

def process_video_generation(
    text: str,
    provider: str,
    voice_name: str,
    model_id: str,
    images: List[gr.File],
    max_workers: int,
    progress=gr.Progress()
) -> Tuple[Optional[str], str, str]:
    """
    Processa geração completa de vídeo

    Args:
        text: Texto de entrada
        provider: Provedor de áudio ('elevenlabs' ou 'minimax')
        voice_name: Nome da voz selecionada
        model_id: Modelo a usar (relevante para ElevenLabs)
        images: Lista de imagens enviadas
        max_workers: Número de requisições simultâneas para WaveSpeed
        progress: Objeto de progresso do Gradio

    Returns:
        (video_path, status_message, error_message)
    """
    try:
        # Valida inputs
        if not text or not text.strip():
            return None, "", "Por favor, digite o roteiro do vídeo"

        if not images or len(images) == 0:
            return None, "", "Por favor, faça upload de pelo menos uma imagem"

        # Extrai paths das imagens
        image_paths = []
        for img in images:
            if hasattr(img, 'name'):
                image_paths.append(img.name)
            elif isinstance(img, str):
                image_paths.append(img)
            else:
                logger.warning(f"Formato de imagem desconhecido: {type(img)}")

        if not image_paths:
            return None, "", "Não foi possível processar as imagens enviadas"

        logger.info(f"Iniciando processamento com {len(image_paths)} imagens e provedor {provider}")

        # Cria job manager com provedor escolhido
        progress(0, desc="Criando job...")
        job_mgr = JobManager(audio_provider=provider)

        job, error = job_mgr.create_job(
            input_text=text,
            voice_name=voice_name,
            image_paths=image_paths,
            model_id=model_id
        )

        if error:
            return None, "", f"Erro na validação: {error}"

        # Processa job com max_workers configurável
        def update_gradio_progress(message: str, percent: int):
            """Callback para atualizar progresso no Gradio"""
            progress(percent / 100, desc=message)

        final_video = job_mgr.process_job(
            job=job,
            progress_callback=update_gradio_progress,
            max_workers_video=max_workers
        )

        # Retorna vídeo gerado
        success_message = f"""
**Vídeo gerado com sucesso!**

Job ID: `{job.job_id}`
Localização: `{final_video}`
Processado em: {(job.completed_at - job.created_at).total_seconds():.1f} segundos

Você pode fazer download do vídeo abaixo.
"""

        logger.info(f"Job {job.job_id} concluído com sucesso")

        return str(final_video), success_message, ""

    except Exception as e:
        error_msg = f"Erro durante processamento: {str(e)}"
        logger.error(error_msg)
        return None, "", error_msg

def process_multiple_scripts(
    scripts_file: gr.File,
    provider: str,
    voice_name: str,
    model_id: str,
    images: List[gr.File],
    max_workers: int,
    progress=gr.Progress()
) -> Tuple[Optional[str], str, str]:
    """
    Processa múltiplos roteiros em sequência

    Args:
        scripts_file: Arquivo de texto com múltiplos roteiros separados por "---"
        provider: Provedor de áudio ('elevenlabs' ou 'minimax')
        voice_name: Nome da voz selecionada
        model_id: Modelo a usar (relevante para ElevenLabs)
        images: Lista de imagens enviadas
        max_workers: Número de requisições simultâneas para WaveSpeed
        progress: Objeto de progresso do Gradio

    Returns:
        (ultimo_video, status_message, error_message)
    """
    try:
        # Valida arquivo
        if not scripts_file:
            return None, "", "Por favor, faça upload do arquivo com os roteiros"

        if not images or len(images) == 0:
            return None, "", "Por favor, faça upload de pelo menos uma imagem"

        # Lê arquivo e separa roteiros
        file_path = scripts_file.name if hasattr(scripts_file, 'name') else scripts_file

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Separa roteiros por "---"
        scripts = [s.strip() for s in content.split('---') if s.strip()]

        if not scripts:
            return None, "", "Nenhum roteiro encontrado. Separe os roteiros com '---'"

        logger.info(f"Encontrados {len(scripts)} roteiros para processar")

        # Extrai paths das imagens
        image_paths = []
        for img in images:
            if hasattr(img, 'name'):
                image_paths.append(img.name)
            elif isinstance(img, str):
                image_paths.append(img)

        # Cria job manager com provedor escolhido
        job_mgr = JobManager(audio_provider=provider)

        # Processa cada roteiro
        results = []
        videos_gerados = []

        for idx, script in enumerate(scripts, 1):
            try:
                progress((idx - 1) / len(scripts), desc=f"Processando roteiro {idx}/{len(scripts)}...")

                logger.info(f"Iniciando roteiro {idx}/{len(scripts)} com provedor {provider}")

                # Cria job
                job, error = job_mgr.create_job(
                    input_text=script,
                    voice_name=voice_name,
                    image_paths=image_paths,
                    model_id=model_id
                )

                if error:
                    results.append(f"Roteiro {idx}: Erro na validação - {error}")
                    continue

                # Processa job
                def update_progress(message: str, percent: int):
                    base_progress = (idx - 1) / len(scripts)
                    current_progress = base_progress + (percent / 100) / len(scripts)
                    progress(current_progress, desc=f"Roteiro {idx}/{len(scripts)}: {message}")

                final_video = job_mgr.process_job(
                    job=job,
                    progress_callback=update_progress,
                    max_workers_video=max_workers
                )

                videos_gerados.append(final_video)
                duration = (job.completed_at - job.created_at).total_seconds()
                results.append(f"Roteiro {idx}: Concluído em {duration:.1f}s - {final_video}")
                logger.info(f"Roteiro {idx} concluído: {final_video}")

            except Exception as e:
                results.append(f"Roteiro {idx}: Erro - {str(e)}")
                logger.error(f"Erro no roteiro {idx}: {e}")
                continue

        # Resultado final
        success_count = len(videos_gerados)
        total_count = len(scripts)

        results_text = chr(10).join(results)
        status_message = f"""
**Processamento em Lote Concluído**

**Resumo:**
- Total de roteiros: {total_count}
- Vídeos gerados: {success_count}
- Falhas: {total_count - success_count}

**Resultados:**

{results_text}

Todos os vídeos foram salvos em suas respectivas pastas temp/job_*
"""

        # Retorna o último vídeo gerado (se houver)
        ultimo_video = videos_gerados[-1] if videos_gerados else None

        return str(ultimo_video) if ultimo_video else None, status_message, ""

    except Exception as e:
        error_msg = f"Erro durante processamento em lote: {str(e)}"
        logger.error(error_msg)
        return None, "", error_msg


# ============================================================================
# NOVAS FUNÇÕES PARA MÚLTIPLOS ROTEIROS COM PREVIEW
# ============================================================================

def parse_scripts_for_preview(scripts_text: str) -> Tuple[List[Dict], str]:
    """
    Analisa o texto com múltiplos roteiros e retorna a estrutura para preview

    Args:
        scripts_text: Texto com roteiros separados por ---

    Returns:
        (lista_de_scripts_com_batches, mensagem_status)
    """
    if not scripts_text or not scripts_text.strip():
        return [], "Digite os roteiros separados por '---'"

    # Separa roteiros por "---"
    raw_scripts = [s.strip() for s in scripts_text.split('---') if s.strip()]

    if not raw_scripts:
        return [], "Nenhum roteiro encontrado. Separe os roteiros com '---'"

    scripts_data = []

    for idx, script in enumerate(raw_scripts, 1):
        # Divide em parágrafos
        paragraphs = split_into_paragraphs(script)

        # Cria batches
        batches = create_batches(paragraphs, Config.BATCH_SIZE)

        # Monta estrutura do roteiro
        script_data = {
            "id": idx,
            "text": script,
            "paragraphs": paragraphs,
            "batches": [
                {
                    "batch_number": b_idx + 1,
                    "text": "\n\n".join(batch),
                    "char_count": sum(len(p) for p in batch),
                    "image_index": 0  # Índice da imagem selecionada (0 = aleatório)
                }
                for b_idx, batch in enumerate(batches)
            ],
            "voice": None,  # Será selecionado pelo usuário
            "total_chars": len(script),
            "total_batches": len(batches)
        }

        scripts_data.append(script_data)

    total_batches = sum(s["total_batches"] for s in scripts_data)
    total_chars = sum(s["total_chars"] for s in scripts_data)

    status = f"""**Preview Gerado com Sucesso!**

- **Roteiros encontrados:** {len(scripts_data)}
- **Total de batches:** {total_batches}
- **Total de caracteres:** {total_chars:,}

Agora você pode:
1. Selecionar a voz para cada roteiro
2. Escolher a imagem para cada batch
3. Clicar em "Processar Todos" para gerar os vídeos
"""

    return scripts_data, status


def generate_preview_html(scripts_data: List[Dict], images: List, provider: str) -> Tuple[str, Any]:
    """
    Gera o HTML do preview dos roteiros com batches

    Args:
        scripts_data: Lista de dados dos roteiros
        images: Lista de imagens carregadas
        provider: Provedor de áudio selecionado

    Returns:
        (html_content, state_data)
    """
    if not scripts_data:
        return "<div style='padding: 20px; text-align: center; color: #666;'>Nenhum roteiro para exibir. Digite os roteiros e clique em 'Gerar Preview'.</div>", None

    # Obtém nomes das imagens
    image_names = ["Aleatório"]
    if images:
        for idx, img in enumerate(images, 1):
            if hasattr(img, 'name'):
                name = Path(img.name).name
            else:
                name = f"Imagem {idx}"
            image_names.append(name)

    # Obtém vozes disponíveis
    voices = get_voice_choices(provider)

    # CSS do preview
    css = """
    <style>
        .preview-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 100%;
            padding: 10px;
        }
        .script-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            margin-bottom: 20px;
            padding: 20px;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .script-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.3);
        }
        .script-title {
            font-size: 18px;
            font-weight: bold;
        }
        .script-stats {
            font-size: 12px;
            opacity: 0.9;
        }
        .batch-container {
            background: rgba(255,255,255,0.95);
            border-radius: 8px;
            margin-top: 10px;
            padding: 15px;
            color: #333;
        }
        .batch-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid #eee;
        }
        .batch-title {
            font-weight: 600;
            color: #667eea;
        }
        .batch-text {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            max-height: 150px;
            overflow-y: auto;
            margin-bottom: 10px;
        }
        .batch-controls {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }
        .control-label {
            font-size: 12px;
            color: #666;
            margin-right: 5px;
        }
        .info-badge {
            background: #e8f4fd;
            color: #1976d2;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
        }
    </style>
    """

    # Gera HTML para cada roteiro
    html_parts = [css, "<div class='preview-container'>"]

    for script in scripts_data:
        html_parts.append(f"""
        <div class='script-card' data-script-id='{script["id"]}'>
            <div class='script-header'>
                <div>
                    <div class='script-title'>Roteiro #{script["id"]}</div>
                    <div class='script-stats'>{script["total_chars"]:,} caracteres | {script["total_batches"]} batches</div>
                </div>
                <div class='info-badge'>Selecione a voz abaixo</div>
            </div>
        """)

        # Adiciona batches
        for batch in script["batches"]:
            preview_text = batch["text"][:300] + "..." if len(batch["text"]) > 300 else batch["text"]

            html_parts.append(f"""
            <div class='batch-container' data-batch-id='{batch["batch_number"]}'>
                <div class='batch-header'>
                    <span class='batch-title'>Batch #{batch["batch_number"]}</span>
                    <span class='info-badge'>{batch["char_count"]} caracteres</span>
                </div>
                <div class='batch-text'>{preview_text}</div>
                <div class='batch-controls'>
                    <span class='control-label'>Imagem selecionada via dropdown abaixo</span>
                </div>
            </div>
            """)

        html_parts.append("</div>")

    html_parts.append("</div>")

    # Prepara state data
    state_data = {
        "scripts": scripts_data,
        "image_names": image_names,
        "voices": voices
    }

    return "\n".join(html_parts), state_data


def create_preview_and_update(
    scripts_text: str,
    images: List,
    provider: str
) -> Tuple[str, str, Any, Any, Any]:
    """
    Cria preview dos roteiros e atualiza a interface

    Returns:
        (preview_html, status_msg, state, voice_updates, image_updates)
    """
    scripts_data, status = parse_scripts_for_preview(scripts_text)

    if not scripts_data:
        empty_html = "<div style='padding: 20px; text-align: center; color: #666;'>Nenhum roteiro para exibir.</div>"
        return empty_html, status, None, [], []

    preview_html, state_data = generate_preview_html(scripts_data, images, provider)

    # Prepara dados para os dropdowns dinâmicos
    voices = get_voice_choices(provider)
    image_names = ["Aleatório (automático)"]
    if images:
        for idx, img in enumerate(images, 1):
            if hasattr(img, 'name'):
                name = Path(img.name).name
            else:
                name = f"Imagem {idx}"
            image_names.append(name)

    # Cria lista de atualizações para vozes (uma por roteiro)
    voice_choices = [(v, v) for v in voices]

    # Cria lista de atualizações para imagens (uma por batch)
    image_choices = [(name, idx) for idx, name in enumerate(image_names)]

    return preview_html, status, state_data, voice_choices, image_choices


def process_preview_scripts(
    state_data: Dict,
    provider: str,
    model_id: str,
    images: List,
    max_workers: int,
    voice_selections: List[str],
    image_selections: List[List[int]],
    progress=gr.Progress()
) -> Tuple[Optional[str], str, str]:
    """
    Processa os roteiros do preview com as configurações selecionadas

    Args:
        state_data: Dados do estado com os roteiros
        provider: Provedor de áudio
        model_id: Modelo de voz
        images: Lista de imagens
        max_workers: Workers simultâneos
        voice_selections: Lista de vozes selecionadas (uma por roteiro)
        image_selections: Lista de listas de índices de imagem (por roteiro > por batch)
        progress: Objeto de progresso

    Returns:
        (ultimo_video, status, erro)
    """
    try:
        if not state_data or "scripts" not in state_data:
            return None, "", "Nenhum roteiro no preview. Gere o preview primeiro."

        scripts = state_data["scripts"]

        if not scripts:
            return None, "", "Nenhum roteiro para processar."

        if not images or len(images) == 0:
            return None, "", "Por favor, faça upload de pelo menos uma imagem"

        # Extrai paths das imagens
        image_paths = []
        for img in images:
            if hasattr(img, 'name'):
                image_paths.append(img.name)
            elif isinstance(img, str):
                image_paths.append(img)

        if not image_paths:
            return None, "", "Não foi possível processar as imagens enviadas"

        # Cria job manager
        job_mgr = JobManager(audio_provider=provider)

        results = []
        videos_gerados = []

        total_scripts = len(scripts)

        for idx, script_data in enumerate(scripts):
            try:
                script_num = script_data["id"]
                script_text = script_data["text"]

                # Obtém a voz selecionada para este roteiro
                voice_name = voice_selections[idx] if idx < len(voice_selections) else voice_selections[0]

                progress((idx) / total_scripts, desc=f"Processando roteiro {script_num}/{total_scripts}...")

                logger.info(f"Processando roteiro {script_num} com voz: {voice_name}")

                # Para seleção de imagem por batch, precisamos modificar o job_manager
                # Por agora, usamos a imagem selecionada para o primeiro batch como padrão
                # ou mantemos aleatório se 0

                # Cria job
                job, error = job_mgr.create_job(
                    input_text=script_text,
                    voice_name=voice_name,
                    image_paths=image_paths,
                    model_id=model_id
                )

                if error:
                    results.append(f"[X] Roteiro {script_num}: Erro - {error}")
                    continue

                # Processa job
                def update_progress(message: str, percent: int):
                    base = idx / total_scripts
                    current = base + (percent / 100) / total_scripts
                    progress(current, desc=f"Roteiro {script_num}: {message}")

                final_video = job_mgr.process_job(
                    job=job,
                    progress_callback=update_progress,
                    max_workers_video=max_workers
                )

                videos_gerados.append(str(final_video))
                duration = (job.completed_at - job.created_at).total_seconds()
                results.append(f"[OK] Roteiro {script_num}: Concluído em {duration:.1f}s")
                logger.info(f"Roteiro {script_num} concluído: {final_video}")

            except Exception as e:
                results.append(f"[X] Roteiro {script_data['id']}: Erro - {str(e)}")
                logger.error(f"Erro no roteiro {script_data['id']}: {e}")
                continue

        progress(1.0, desc="Processamento concluído!")

        # Resultado final
        success_count = len(videos_gerados)

        status_message = f"""
**Processamento Concluído!**

**Resumo:**
- Roteiros processados: {success_count}/{total_scripts}
- Falhas: {total_scripts - success_count}

**Detalhes:**
{chr(10).join(results)}

**Vídeos gerados:**
{chr(10).join(f"- {v}" for v in videos_gerados) if videos_gerados else "Nenhum vídeo gerado"}
"""

        ultimo_video = videos_gerados[-1] if videos_gerados else None

        return ultimo_video, status_message, ""

    except Exception as e:
        error_msg = f"Erro durante processamento: {str(e)}"
        logger.error(error_msg)
        return None, "", error_msg


def create_interface():
    """Cria interface Gradio"""

    # Tema customizado
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
    )

    with gr.Blocks(theme=theme, title="Gerador de Vídeos com Lip-Sync") as app:

        gr.Markdown("""
        # Gerador de Vídeos com Lip-Sync

        Transforme seu roteiro em vídeos profissionais com sincronização labial automática!

        ### Modos disponíveis:
        - **Vídeo Único**: Processa um roteiro por vez
        - **Processamento em Lote**: Processa múltiplos roteiros sequencialmente
        - **Múltiplos Roteiros com Preview**: Adicione vários roteiros, visualize batches e escolha voz/imagem para cada um

        ---
        """)

        with gr.Tabs():
            # TAB 1: Vídeo Único
            with gr.Tab("Vídeo Único"):
                with gr.Row():
                    with gr.Column(scale=2):

                        # INPUT: Texto
                        text_input = gr.Textbox(
                            label="Roteiro do Vídeo",
                            placeholder="Digite ou cole o texto completo do seu roteiro aqui...\n\nCada parágrafo será processado separadamente.",
                            lines=15,
                            max_lines=30
                        )

                        # INPUT: Provedor de Áudio
                        provider_dropdown = gr.Dropdown(
                            label="Provedor de Áudio",
                            choices=get_audio_provider_choices(),
                            value=get_audio_provider_choices()[0][1] if get_audio_provider_choices() else "elevenlabs",
                            info="Escolha o serviço de síntese de voz"
                        )

                        # INPUT: Voz
                        voice_dropdown = gr.Dropdown(
                            label="Selecione a Voz",
                            choices=[],
                            value=None,
                            info="As vozes serão carregadas de acordo com o provedor selecionado"
                        )

                        # INPUT: Modelo
                        model_dropdown = gr.Dropdown(
                            label="Selecione o Modelo de Voz (ElevenLabs)",
                            choices=get_model_choices(),
                            value="eleven_multilingual_v3",
                            visible=True,
                            info="Relevante apenas para ElevenLabs"
                        )

                        # INPUT: Imagens
                        images_input = gr.File(
                            label="Imagens do Apresentador (1-20 imagens PNG/JPG)",
                            file_count="multiple",
                            file_types=["image"]
                        )

                        # INPUT: Max Workers
                        max_workers_single = gr.Slider(
                            label="Vídeos Simultâneos no WaveSpeed",
                            info="Quantos vídeos processar ao mesmo tempo no WaveSpeed (mais = mais rápido, mas usa mais créditos)",
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1
                        )

                        # Botão de estimativa
                        estimate_btn = gr.Button("Estimar Custo e Tempo", variant="secondary", size="sm")

                        # Área de estimativa
                        estimate_output = gr.Markdown(label="Estimativa")

                        # Botão de processar
                        process_btn = gr.Button("Gerar Vídeo", variant="primary", size="lg")

                    with gr.Column(scale=1):

                        gr.Markdown("### Status do Processamento")

                        # Mensagem de sucesso
                        status_output = gr.Markdown(label="Status")

                        # Mensagem de erro
                        error_output = gr.Markdown(label="Erro")

                        # OUTPUT: Vídeo final
                        video_output = gr.Video(
                            label="Vídeo Final",
                            format="mp4"
                        )

            # TAB 2: Processamento em Lote (mantido para compatibilidade)
            with gr.Tab("Processamento em Lote"):
                gr.Markdown("""
                ### Como usar o processamento em lote:

                1. Crie um arquivo `.txt` com múltiplos roteiros
                2. Separe cada roteiro com uma linha contendo apenas `---`
                3. Faça upload do arquivo abaixo
                4. Configure as opções e clique em "Processar Lote"

                **Exemplo de arquivo:**
                ```
                Olá! Este é o primeiro roteiro.
                Ele será processado primeiro.
                ---
                Este é o segundo roteiro.
                Será processado após o primeiro.
                ---
                E este é o terceiro roteiro.
                ```
                """)

                with gr.Row():
                    with gr.Column(scale=2):

                        # INPUT: Arquivo de roteiros
                        scripts_file_input = gr.File(
                            label="Arquivo de Roteiros (.txt)",
                            file_types=[".txt"],
                            file_count="single"
                        )

                        # INPUT: Provedor de Áudio (batch)
                        provider_dropdown_batch = gr.Dropdown(
                            label="Provedor de Áudio",
                            choices=get_audio_provider_choices(),
                            value=get_audio_provider_choices()[0][1] if get_audio_provider_choices() else "elevenlabs",
                            info="Escolha o serviço de síntese de voz"
                        )

                        # INPUT: Voz (batch)
                        voice_dropdown_batch = gr.Dropdown(
                            label="Selecione a Voz",
                            choices=[],
                            value=None,
                            info="As vozes serão carregadas de acordo com o provedor selecionado"
                        )

                        # INPUT: Modelo (batch)
                        model_dropdown_batch = gr.Dropdown(
                            label="Selecione o Modelo de Voz (ElevenLabs)",
                            choices=get_model_choices(),
                            value="eleven_multilingual_v3",
                            info="Relevante apenas para ElevenLabs"
                        )

                        # INPUT: Imagens (batch)
                        images_input_batch = gr.File(
                            label="Imagens do Apresentador (1-20 imagens PNG/JPG)",
                            file_count="multiple",
                            file_types=["image"]
                        )

                        # INPUT: Max Workers (batch)
                        max_workers_batch = gr.Slider(
                            label="Vídeos Simultâneos no WaveSpeed (por roteiro)",
                            info="Quantos vídeos processar ao mesmo tempo no WaveSpeed para CADA roteiro",
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1
                        )

                        # Botão de processar lote
                        process_batch_btn = gr.Button("Processar Lote", variant="primary", size="lg")

                    with gr.Column(scale=1):

                        gr.Markdown("### Status do Processamento")

                        # Mensagem de sucesso (batch)
                        status_output_batch = gr.Markdown(label="Status")

                        # Mensagem de erro (batch)
                        error_output_batch = gr.Markdown(label="Erro")

                        # OUTPUT: Último vídeo gerado
                        video_output_batch = gr.Video(
                            label="Último Vídeo Gerado",
                            format="mp4"
                        )

            # TAB 3: NOVA - Múltiplos Roteiros com Preview
            with gr.Tab("Múltiplos Roteiros com Preview"):
                gr.Markdown("""
                ### Adicione múltiplos roteiros com preview avançado

                **Como usar:**
                1. Cole seus roteiros separados por `---`
                2. Faça upload das imagens que serão usadas
                3. Clique em **"Gerar Preview"** para visualizar
                4. No preview, selecione a **voz para cada roteiro**
                5. Escolha a **imagem para cada batch** (ou deixe aleatório)
                6. Clique em **"Processar Todos"** para gerar os vídeos
                """)

                # Estado para armazenar dados do preview
                preview_state_var = gr.State(value=None)

                with gr.Row():
                    # Coluna esquerda - Inputs
                    with gr.Column(scale=1):
                        gr.Markdown("#### 1. Digite os Roteiros")

                        # INPUT: Texto com múltiplos roteiros
                        multi_scripts_input = gr.Textbox(
                            label="Roteiros (separe com ---)",
                            placeholder="""Roteiro 1: Olá! Bem-vindo ao nosso canal.
Hoje vamos falar sobre tecnologia.

Este é o segundo parágrafo do primeiro roteiro.
---
Roteiro 2: Este é outro roteiro completamente diferente.

Com seus próprios parágrafos e conteúdo.

E mais um parágrafo aqui.
---
Roteiro 3: E aqui está o terceiro roteiro.

Cada um será processado separadamente com suas próprias configurações!""",
                            lines=15,
                            max_lines=40
                        )

                        gr.Markdown("#### 2. Upload das Imagens")

                        # INPUT: Imagens
                        images_preview_input = gr.File(
                            label="Imagens do Apresentador (1-20 imagens PNG/JPG)",
                            file_count="multiple",
                            file_types=["image"]
                        )

                        gr.Markdown("#### 3. Configurações Gerais")

                        # INPUT: Provedor de Áudio
                        provider_preview = gr.Dropdown(
                            label="Provedor de Áudio",
                            choices=get_audio_provider_choices(),
                            value=get_audio_provider_choices()[0][1] if get_audio_provider_choices() else "elevenlabs",
                            info="Serviço de síntese de voz"
                        )

                        # INPUT: Modelo
                        model_preview = gr.Dropdown(
                            label="Modelo de Voz (ElevenLabs)",
                            choices=get_model_choices(),
                            value="eleven_multilingual_v3"
                        )

                        # INPUT: Max Workers
                        max_workers_preview = gr.Slider(
                            label="Vídeos Simultâneos no WaveSpeed",
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1
                        )

                        # Botão para gerar preview
                        generate_preview_btn = gr.Button(
                            "Gerar Preview dos Roteiros",
                            variant="secondary",
                            size="lg"
                        )

                        # Status do preview
                        preview_status = gr.Markdown("")

                    # Coluna direita - Preview e Configurações por roteiro
                    with gr.Column(scale=2):
                        gr.Markdown("#### 4. Preview e Configurações")

                        # Área de preview HTML
                        preview_html = gr.HTML(
                            value="<div style='padding: 40px; text-align: center; color: #666; background: #f5f5f5; border-radius: 12px;'>"
                                  "<h3>Preview dos Roteiros</h3>"
                                  "<p>Digite os roteiros, faça upload das imagens e clique em 'Gerar Preview'</p></div>"
                        )

                        gr.Markdown("#### 5. Seleção de Voz por Roteiro")
                        gr.Markdown("*Após gerar o preview, selecione a voz para cada roteiro:*")

                        # Dropdowns dinâmicos para vozes (máximo 10 roteiros)
                        voice_selections = []
                        with gr.Row():
                            with gr.Column():
                                for i in range(5):
                                    v = gr.Dropdown(
                                        label=f"Voz - Roteiro {i+1}",
                                        choices=[],
                                        visible=False,
                                        interactive=True
                                    )
                                    voice_selections.append(v)
                            with gr.Column():
                                for i in range(5, 10):
                                    v = gr.Dropdown(
                                        label=f"Voz - Roteiro {i+1}",
                                        choices=[],
                                        visible=False,
                                        interactive=True
                                    )
                                    voice_selections.append(v)

                        gr.Markdown("#### 6. Seleção de Imagem por Batch")
                        gr.Markdown("*Selecione qual imagem usar em cada batch (deixe 'Aleatório' para seleção automática):*")

                        # Dropdowns para seleção de imagem por batch (layout simplificado)
                        # Máximo de 20 batches por interface
                        image_selections = []
                        with gr.Accordion("Configuração de Imagens por Batch", open=False):
                            for script_idx in range(5):  # Até 5 roteiros visíveis
                                with gr.Group():
                                    gr.Markdown(f"**Roteiro {script_idx + 1}**")
                                    batch_row = []
                                    with gr.Row():
                                        for batch_idx in range(4):  # Até 4 batches por roteiro
                                            img_dd = gr.Dropdown(
                                                label=f"Batch {batch_idx + 1}",
                                                choices=["Aleatório"],
                                                value="Aleatório",
                                                visible=False,
                                                scale=1
                                            )
                                            batch_row.append(img_dd)
                                    image_selections.append(batch_row)

                        # Botão para processar
                        process_preview_btn = gr.Button(
                            "Processar Todos os Roteiros",
                            variant="primary",
                            size="lg"
                        )

                        gr.Markdown("#### Resultado")

                        # Status do processamento
                        process_status_preview = gr.Markdown("")

                        # Erro
                        error_preview = gr.Markdown("")

                        # Vídeo de saída
                        video_output_preview = gr.Video(
                            label="Último Vídeo Gerado",
                            format="mp4"
                        )

        # Informações adicionais
        with gr.Accordion("Informações Técnicas", open=False):
            gr.Markdown(f"""
            ### Configurações Atuais

            **APIs Integradas:**
            - **Gemini 2.5 Flash Lite**: Formatação e otimização de texto
            - **ElevenLabs**: Síntese de voz de alta qualidade
            - **WaveSpeed Wan 2.2**: Geração de vídeo com lip-sync
            - **FFmpeg**: Concatenação de vídeos

            **Limites:**
            - Texto: {Config.MIN_TEXT_LENGTH:,} - {Config.MAX_TEXT_LENGTH:,} caracteres
            - Imagens: {Config.MIN_IMAGES} - {Config.MAX_IMAGES} arquivos
            - Formatos de imagem: PNG, JPG, JPEG
            - Resolução de vídeo: {Config.DEFAULT_RESOLUTION}
            - Batches de texto: {Config.BATCH_SIZE} parágrafos por batch

            **Estrutura de Pastas:**
            - Diretório temporário: `{Config.TEMP_FOLDER}`
            - Cada job cria: `/job_{{uuid}}/{{formatted_text, audios, videos, images}}/`
            - Vídeo final: `final_output.mp4`
            """)

        with gr.Accordion("Exemplo de Roteiro", open=False):
            example_script = """Olá! Bem-vindo ao nosso canal sobre tecnologia e inovação.

Hoje vamos falar sobre inteligência artificial e como ela está transformando o mundo dos negócios.

A IA não é mais ficção científica. Ela está presente em nossas vidas diariamente, desde assistentes virtuais até sistemas de recomendação.

Empresas de todos os tamanhos estão adotando IA para automatizar processos, melhorar a experiência do cliente e tomar decisões mais inteligentes.

Neste vídeo, você vai aprender os conceitos básicos de IA, suas aplicações práticas e como começar a implementar em sua empresa.

Fique conosco até o final para descobrir as tendências que vão dominar o mercado nos próximos anos!

Não se esqueça de se inscrever no canal e ativar o sininho para não perder nenhuma novidade.

Vamos começar!"""

            gr.Textbox(
                value=example_script,
                label="Exemplo de roteiro formatado",
                lines=10,
                interactive=False
            )

        # ============================================
        # EVENTOS - Vídeo Único
        # ============================================
        provider_dropdown.change(
            fn=update_voices_by_provider,
            inputs=[provider_dropdown],
            outputs=[voice_dropdown]
        )

        estimate_btn.click(
            fn=estimate_job,
            inputs=[text_input],
            outputs=[estimate_output]
        )

        process_btn.click(
            fn=process_video_generation,
            inputs=[text_input, provider_dropdown, voice_dropdown, model_dropdown, images_input, max_workers_single],
            outputs=[video_output, status_output, error_output]
        )

        # ============================================
        # EVENTOS - Processamento em Lote
        # ============================================
        provider_dropdown_batch.change(
            fn=update_voices_by_provider,
            inputs=[provider_dropdown_batch],
            outputs=[voice_dropdown_batch]
        )

        process_batch_btn.click(
            fn=process_multiple_scripts,
            inputs=[scripts_file_input, provider_dropdown_batch, voice_dropdown_batch, model_dropdown_batch, images_input_batch, max_workers_batch],
            outputs=[video_output_batch, status_output_batch, error_output_batch]
        )

        # ============================================
        # EVENTOS - Múltiplos Roteiros com Preview
        # ============================================

        def on_generate_preview(scripts_text, images, provider):
            """Handler para gerar preview"""
            scripts_data, status = parse_scripts_for_preview(scripts_text)

            if not scripts_data:
                empty_html = "<div style='padding: 40px; text-align: center; color: #666; background: #f5f5f5; border-radius: 12px;'><p>" + status + "</p></div>"
                # Retorna updates vazios para todos os componentes
                updates = [empty_html, status, None]
                # Voice dropdowns - ocultar todos
                for _ in range(10):
                    updates.append(gr.Dropdown(visible=False, choices=[]))
                # Image dropdowns - ocultar todos
                for _ in range(5):
                    for _ in range(4):
                        updates.append(gr.Dropdown(visible=False, choices=["Aleatório"]))
                return updates

            # Gera HTML do preview
            preview_html_content, state_data = generate_preview_html(scripts_data, images, provider)

            # Obtém vozes
            voices = get_voice_choices(provider)

            # Obtém nomes das imagens
            image_names = ["Aleatório"]
            if images:
                for idx, img in enumerate(images, 1):
                    if hasattr(img, 'name'):
                        name = Path(img.name).name
                    else:
                        name = f"Imagem {idx}"
                    image_names.append(name)

            # Prepara updates
            updates = [preview_html_content, status, state_data]

            # Atualiza dropdowns de voz
            num_scripts = len(scripts_data)
            for i in range(10):
                if i < num_scripts:
                    updates.append(gr.Dropdown(
                        choices=voices,
                        value=voices[0] if voices else None,
                        visible=True,
                        label=f"Voz - Roteiro {i+1}"
                    ))
                else:
                    updates.append(gr.Dropdown(visible=False, choices=[]))

            # Atualiza dropdowns de imagem
            for script_idx in range(5):
                if script_idx < num_scripts:
                    num_batches = scripts_data[script_idx]["total_batches"]
                    for batch_idx in range(4):
                        if batch_idx < num_batches:
                            updates.append(gr.Dropdown(
                                choices=image_names,
                                value="Aleatório",
                                visible=True,
                                label=f"R{script_idx+1} - Batch {batch_idx+1}"
                            ))
                        else:
                            updates.append(gr.Dropdown(visible=False, choices=["Aleatório"]))
                else:
                    for _ in range(4):
                        updates.append(gr.Dropdown(visible=False, choices=["Aleatório"]))

            return updates

        # Flatten image_selections para outputs
        image_selection_flat = []
        for row in image_selections:
            image_selection_flat.extend(row)

        generate_preview_btn.click(
            fn=on_generate_preview,
            inputs=[multi_scripts_input, images_preview_input, provider_preview],
            outputs=[preview_html, preview_status, preview_state_var] + voice_selections + image_selection_flat
        )

        # Atualiza vozes quando provider muda
        def update_preview_voices(provider):
            voices = get_voice_choices(provider)
            updates = []
            for i in range(10):
                updates.append(gr.Dropdown(choices=voices, value=voices[0] if voices else None))
            return updates

        provider_preview.change(
            fn=update_preview_voices,
            inputs=[provider_preview],
            outputs=voice_selections
        )

        def on_process_preview(state_data, provider, model_id, images, max_workers, *voice_and_image_args):
            """Handler para processar roteiros do preview"""
            # Separa argumentos de voz e imagem
            voice_args = list(voice_and_image_args[:10])
            image_args = list(voice_and_image_args[10:])

            # Filtra vozes válidas
            valid_voices = [v for v in voice_args if v]

            if not valid_voices:
                return None, "", "Selecione pelo menos uma voz para os roteiros"

            return process_preview_scripts(
                state_data=state_data,
                provider=provider,
                model_id=model_id,
                images=images,
                max_workers=max_workers,
                voice_selections=valid_voices,
                image_selections=image_args
            )

        process_preview_btn.click(
            fn=on_process_preview,
            inputs=[preview_state_var, provider_preview, model_preview, images_preview_input, max_workers_preview] + voice_selections + image_selection_flat,
            outputs=[video_output_preview, process_status_preview, error_preview]
        )

        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666;">
            <p>Desenvolvido com Gradio | Powered by Gemini, ElevenLabs & WaveSpeed</p>
        </div>
        """)

    return app

def main():
    """Função principal"""
    logger.info("Iniciando aplicação Gradio...")

    try:
        # Cria interface
        app = create_interface()

        # Lança aplicação
        logger.info("Abrindo navegador...")
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            inbrowser=True  # Abre automaticamente no navegador padrão
        )
    except Exception as e:
        logger.error(f"Erro ao iniciar aplicação: {e}")
        print(f"\n Erro ao iniciar aplicação: {e}")
        print("\nVerifique:")
        print("1. Se todas as API keys estão configuradas corretamente no .env")
        print("2. Se as dependências estão instaladas: pip install -r requirements.txt")
        print("3. Se o FFmpeg está instalado e no PATH")
        raise

if __name__ == "__main__":
    main()
