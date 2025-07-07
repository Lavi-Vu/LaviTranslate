import gradio as gr
from api import *
from soni_translate.languages_gui import language_data, news
import copy
import logging
import json
from pydub import AudioSegment
from voice_main import ClassVoices
import argparse
import time
import hashlib
import sys

directories = [
    "downloads",
    "logs",
    "weights",
    "clean_song_output",
    "_XTTS_",
    f"audio2{os.sep}audio",
    "audio",
    "outputs",
]
[
    os.makedirs(directory)
    for directory in directories
    if not os.path.exists(directory)
]

class TTS_Info:
    def __init__(self, piper_enabled, xtts_enabled):
        self.list_edge = edge_tts_voices_list()
        self.list_bark = list(BARK_VOICES_LIST.keys())
        self.list_vits = list(VITS_VOICES_LIST.keys())
        self.list_openai_tts = OPENAI_TTS_MODELS
        self.piper_enabled = piper_enabled
        self.list_vits_onnx = (
            piper_tts_voices_list() if self.piper_enabled else []
        )
        self.xtts_enabled = xtts_enabled

    def tts_list(self):
        self.list_coqui_xtts = (
            coqui_xtts_voices_list() if self.xtts_enabled else []
        )
        list_tts = self.list_coqui_xtts + sorted(
            self.list_edge
            + self.list_bark
            + self.list_vits
            + self.list_openai_tts
            + self.list_vits_onnx
        )
        return list_tts


def prog_disp(msg, percent, is_gui, progress=None):
    logger.info(msg)
    if is_gui:
        progress(percent, desc=msg)


def warn_disp(wrn_lang, is_gui):
    logger.warning(wrn_lang)
    if is_gui:
        gr.Warning(wrn_lang)





title = "<center><strong><font size='7'>📽️ LaviTranslate🈷️</font></strong></center>"


def create_gui(theme, logs_in_gui=False):
    with gr.Blocks(theme=theme) as app:
        gr.Markdown(title)
        gr.Markdown(lg_conf["description"])

        with gr.Tab(lg_conf["tab_translate"]):
            with gr.Row():
                with gr.Column():
                    input_data_type = gr.Dropdown(
                        ["SUBMIT VIDEO", "URL", "Find Video Path"],
                        value="SUBMIT VIDEO",
                        label=lg_conf["video_source"],
                    )

                    def swap_visibility(data_type):
                        if data_type == "URL":
                            return (
                                gr.update(visible=False, value=None),
                                gr.update(visible=True, value=""),
                                gr.update(visible=False, value=""),
                            )
                        elif data_type == "SUBMIT VIDEO":
                            return (
                                gr.update(visible=True, value=None),
                                gr.update(visible=False, value=""),
                                gr.update(visible=False, value=""),
                            )
                        elif data_type == "Find Video Path":
                            return (
                                gr.update(visible=False, value=None),
                                gr.update(visible=False, value=""),
                                gr.update(visible=True, value=""),
                            )

                    video_input = gr.File(
                        label="VIDEO",
                        file_count="multiple",
                        type="filepath",
                    )
                    blink_input = gr.Textbox(
                        visible=False,
                        label=lg_conf["link_label"],
                        info=lg_conf["link_info"],
                        placeholder=lg_conf["link_ph"],
                    )
                    directory_input = gr.Textbox(
                        visible=False,
                        label=lg_conf["dir_label"],
                        info=lg_conf["dir_info"],
                        placeholder=lg_conf["dir_ph"],
                    )
                    input_data_type.change(
                        fn=swap_visibility,
                        inputs=input_data_type,
                        outputs=[video_input, blink_input, directory_input],
                    )

                    gr.HTML()

                    SOURCE_LANGUAGE = gr.Dropdown(
                        LANGUAGES_LIST,
                        value=LANGUAGES_LIST[0],
                        label=lg_conf["sl_label"],
                        info=lg_conf["sl_info"],
                    )
                    TRANSLATE_AUDIO_TO = gr.Dropdown(
                        LANGUAGES_LIST[1:],
                        value="English (en)",
                        label=lg_conf["tat_label"],
                        info=lg_conf["tat_info"],
                    )

                    gr.HTML("<hr></h2>")

                    gr.Markdown(lg_conf["num_speakers"])
                    MAX_TTS = 12
                    min_speakers = gr.Slider(
                        1,
                        MAX_TTS,
                        value=1,
                        label=lg_conf["min_sk"],
                        step=1,
                        visible=False,
                    )
                    max_speakers = gr.Slider(
                        1,
                        MAX_TTS,
                        value=2,
                        step=1,
                        label=lg_conf["max_sk"],
                    )
                    gr.Markdown(lg_conf["tts_select"])

                    def submit(value):
                        visibility_dict = {
                            f"tts_voice{i:02d}": gr.update(visible=i < value)
                            for i in range(MAX_TTS)
                        }
                        return [value for value in visibility_dict.values()]

                    tts_voice00 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-EmmaMultilingualNeural-Female",
                        label=lg_conf["sk1"],
                        visible=True,
                        interactive=True,
                    )
                    tts_voice01 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-AndrewMultilingualNeural-Male",
                        label=lg_conf["sk2"],
                        visible=True,
                        interactive=True,
                    )
                    tts_voice02 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-AvaMultilingualNeural-Female",
                        label=lg_conf["sk3"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice03 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-BrianMultilingualNeural-Male",
                        label=lg_conf["sk4"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice04 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="de-DE-SeraphinaMultilingualNeural-Female",
                        label=lg_conf["sk4"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice05 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="de-DE-FlorianMultilingualNeural-Male",
                        label=lg_conf["sk6"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice06 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="fr-FR-VivienneMultilingualNeural-Female",
                        label=lg_conf["sk7"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice07 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="fr-FR-RemyMultilingualNeural-Male",
                        label=lg_conf["sk8"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice08 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-EmmaMultilingualNeural-Female",
                        label=lg_conf["sk9"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice09 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-AndrewMultilingualNeural-Male",
                        label=lg_conf["sk10"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice10 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-EmmaMultilingualNeural-Female",
                        label=lg_conf["sk11"],
                        visible=False,
                        interactive=True,
                    )
                    tts_voice11 = gr.Dropdown(
                        SoniTr.tts_info.tts_list(),
                        value="en-US-AndrewMultilingualNeural-Male",
                        label=lg_conf["sk12"],
                        visible=False,
                        interactive=True,
                    )
                    max_speakers.change(
                        submit,
                        max_speakers,
                        [
                            tts_voice00,
                            tts_voice01,
                            tts_voice02,
                            tts_voice03,
                            tts_voice04,
                            tts_voice05,
                            tts_voice06,
                            tts_voice07,
                            tts_voice08,
                            tts_voice09,
                            tts_voice10,
                            tts_voice11,
                        ],
                    )

                    with gr.Column():
                        with gr.Accordion(
                            lg_conf["vc_title"],
                            open=False,
                        ):
                            gr.Markdown(lg_conf["vc_subtitle"])
                            voice_imitation_gui = gr.Checkbox(
                                False,
                                label=lg_conf["vc_active_label"],
                                info=lg_conf["vc_active_info"],
                            )
                            openvoice_models = ["openvoice", "openvoice_v2"]
                            voice_imitation_method_options = (
                                ["freevc"] + openvoice_models
                                if SoniTr.tts_info.xtts_enabled
                                else openvoice_models
                            )
                            voice_imitation_method_gui = gr.Dropdown(
                                voice_imitation_method_options,
                                value=voice_imitation_method_options[0],
                                label=lg_conf["vc_method_label"],
                                info=lg_conf["vc_method_info"],
                            )
                            voice_imitation_max_segments_gui = gr.Slider(
                                label=lg_conf["vc_segments_label"],
                                info=lg_conf["vc_segments_info"],
                                value=3,
                                step=1,
                                minimum=1,
                                maximum=10,
                                visible=True,
                                interactive=True,
                            )
                            voice_imitation_vocals_dereverb_gui = gr.Checkbox(
                                False,
                                label=lg_conf["vc_dereverb_label"],
                                info=lg_conf["vc_dereverb_info"],
                            )
                            voice_imitation_remove_previous_gui = gr.Checkbox(
                                True,
                                label=lg_conf["vc_remove_label"],
                                info=lg_conf["vc_remove_info"],
                            )

                    if SoniTr.tts_info.xtts_enabled:
                        with gr.Column():
                            with gr.Accordion(
                                lg_conf["xtts_title"],
                                open=False,
                            ):
                                gr.Markdown(lg_conf["xtts_subtitle"])
                                wav_speaker_file = gr.File(
                                    label=lg_conf["xtts_file_label"]
                                )
                                wav_speaker_name = gr.Textbox(
                                    label=lg_conf["xtts_name_label"],
                                    value="",
                                    info=lg_conf["xtts_name_info"],
                                    placeholder="default_name",
                                    lines=1,
                                )
                                wav_speaker_start = gr.Number(
                                    label="Time audio start",
                                    value=0,
                                    visible=False,
                                )
                                wav_speaker_end = gr.Number(
                                    label="Time audio end",
                                    value=0,
                                    visible=False,
                                )
                                wav_speaker_dir = gr.Textbox(
                                    label="Directory save",
                                    value="_XTTS_",
                                    visible=False,
                                )
                                wav_speaker_dereverb = gr.Checkbox(
                                    True,
                                    label=lg_conf["xtts_dereverb_label"],
                                    info=lg_conf["xtts_dereverb_info"]
                                )
                                wav_speaker_output = gr.HTML()
                                create_xtts_wav = gr.Button(
                                    lg_conf["xtts_button"]
                                )
                                gr.Markdown(lg_conf["xtts_footer"])
                    else:
                        wav_speaker_dereverb = gr.Checkbox(
                            False,
                            label=lg_conf["xtts_dereverb_label"],
                            info=lg_conf["xtts_dereverb_info"],
                            visible=False
                        )

                    with gr.Column():
                        with gr.Accordion(
                            lg_conf["extra_setting"], open=False
                        ):
                            audio_accelerate = gr.Slider(
                                label=lg_conf["acc_max_label"],
                                value=1.9,
                                step=0.1,
                                minimum=1.0,
                                maximum=2.5,
                                visible=True,
                                interactive=True,
                                info=lg_conf["acc_max_info"],
                            )
                            acceleration_rate_regulation_gui = gr.Checkbox(
                                False,
                                label=lg_conf["acc_rate_label"],
                                info=lg_conf["acc_rate_info"],
                            )
                            avoid_overlap_gui = gr.Checkbox(
                                False,
                                label=lg_conf["or_label"],
                                info=lg_conf["or_info"],
                            )

                            gr.HTML("<hr></h2>")

                            audio_mix_options = [
                                "Mixing audio with sidechain compression",
                                "Adjusting volumes and mixing audio",
                            ]
                            AUDIO_MIX = gr.Dropdown(
                                audio_mix_options,
                                value=audio_mix_options[1],
                                label=lg_conf["aud_mix_label"],
                                info=lg_conf["aud_mix_info"],
                            )
                            volume_original_mix = gr.Slider(
                                label=lg_conf["vol_ori"],
                                info="for Adjusting volumes and mixing audio",
                                value=0.25,
                                step=0.05,
                                minimum=0.0,
                                maximum=2.50,
                                visible=True,
                                interactive=True,
                            )
                            volume_translated_mix = gr.Slider(
                                label=lg_conf["vol_tra"],
                                info="for Adjusting volumes and mixing audio",
                                value=1.80,
                                step=0.05,
                                minimum=0.0,
                                maximum=2.50,
                                visible=True,
                                interactive=True,
                            )
                            main_voiceless_track = gr.Checkbox(
                                label=lg_conf["voiceless_tk_label"],
                                info=lg_conf["voiceless_tk_info"],
                                value=True
                            )

                            gr.HTML("<hr></h2>")
                            sub_type_options = [
                                "disable",
                                "srt",
                                "vtt",
                                "ass",
                                "txt",
                                "tsv",
                                "json",
                                "aud",
                            ]

                            sub_type_output = gr.Dropdown(
                                sub_type_options,
                                value=sub_type_options[1],
                                label=lg_conf["sub_type"],
                            )
                            soft_subtitles_to_video_gui = gr.Checkbox(
                                label=lg_conf["soft_subs_label"],
                                info=lg_conf["soft_subs_info"],
                            )
                            burn_subtitles_to_video_gui = gr.Checkbox(
                                label=lg_conf["burn_subs_label"],
                                info=lg_conf["burn_subs_info"],
                            )

                            gr.HTML("<hr></h2>")
                            gr.Markdown(lg_conf["whisper_title"])
                            literalize_numbers_gui = gr.Checkbox(
                                True,
                                label=lg_conf["lnum_label"],
                                info=lg_conf["lnum_info"],
                            )
                            vocal_refinement_gui = gr.Checkbox(
                                False,
                                label=lg_conf["scle_label"],
                                info=lg_conf["scle_info"],
                            )
                            segment_duration_limit_gui = gr.Slider(
                                label=lg_conf["sd_limit_label"],
                                info=lg_conf["sd_limit_info"],
                                value=15,
                                step=1,
                                minimum=1,
                                maximum=30,
                            )
                            whisper_model_default = (
                                "medium"
                                if SoniTr.device == "cuda"
                                else "medium"
                            )

                            WHISPER_MODEL_SIZE = gr.Dropdown(
                                ASR_MODEL_OPTIONS + find_whisper_models(),
                                value=whisper_model_default,
                                label="Whisper ASR model",
                                info=lg_conf["asr_model_info"],
                                allow_custom_value=True,
                            )
                            com_t_opt, com_t_default = (
                                [COMPUTE_TYPE_GPU, "float16"]
                                if SoniTr.device == "cuda"
                                else [COMPUTE_TYPE_CPU, "float32"]
                            )
                            compute_type = gr.Dropdown(
                                com_t_opt,
                                value=com_t_default,
                                label=lg_conf["ctype_label"],
                                info=lg_conf["ctype_info"],
                            )
                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=32,
                                value=8,
                                label=lg_conf["batchz_label"],
                                info=lg_conf["batchz_info"],
                                step=1,
                            )
                            input_srt = gr.File(
                                label=lg_conf["srt_file_label"],
                                file_types=[".srt", ".ass", ".vtt"],
                                height=130,
                            )

                            gr.HTML("<hr></h2>")
                            text_segmentation_options = [
                                "sentence",
                                "word",
                                "character"
                            ]
                            text_segmentation_scale_gui = gr.Dropdown(
                                text_segmentation_options,
                                value=text_segmentation_options[0],
                                label=lg_conf["tsscale_label"],
                                info=lg_conf["tsscale_info"],
                            )
                            divide_text_segments_by_gui = gr.Textbox(
                                label=lg_conf["divide_text_label"],
                                value="",
                                info=lg_conf["divide_text_info"],
                            )

                            gr.HTML("<hr></h2>")
                            pyannote_models_list = list(
                                diarization_models.keys()
                            )
                            diarization_process_dropdown = gr.Dropdown(
                                pyannote_models_list,
                                value=pyannote_models_list[1],
                                label=lg_conf["diarization_label"],
                            )
                            translate_process_dropdown = gr.Dropdown(
                                TRANSLATION_PROCESS_OPTIONS,
                                value=TRANSLATION_PROCESS_OPTIONS[0],
                                label=lg_conf["tr_process_label"],
                            )

                            gr.HTML("<hr></h2>")
                            main_output_type = gr.Dropdown(
                                OUTPUT_TYPE_OPTIONS,
                                value=OUTPUT_TYPE_OPTIONS[0],
                                label=lg_conf["out_type_label"],
                            )
                            VIDEO_OUTPUT_NAME = gr.Textbox(
                                label=lg_conf["out_name_label"],
                                value="",
                                info=lg_conf["out_name_info"],
                            )
                            play_sound_gui = gr.Checkbox(
                                True,
                                label=lg_conf["task_sound_label"],
                                info=lg_conf["task_sound_info"],
                            )
                            enable_cache_gui = gr.Checkbox(
                                True,
                                label=lg_conf["cache_label"],
                                info=lg_conf["cache_info"],
                            )
                            PREVIEW = gr.Checkbox(
                                label="Preview", info=lg_conf["preview_info"]
                            )
                            is_gui_dummy_check = gr.Checkbox(
                                True, visible=False
                            )

                with gr.Column(variant="compact"):
                    edit_sub_check = gr.Checkbox(
                        label=lg_conf["edit_sub_label"],
                        info=lg_conf["edit_sub_info"],
                    )
                    dummy_false_check = gr.Checkbox(
                        False,
                        visible=False,
                    )

                    def visible_component_subs(input_bool):
                        if input_bool:
                            return gr.update(visible=True), gr.update(
                                visible=True
                            )
                        else:
                            return gr.update(visible=False), gr.update(
                                visible=False
                            )

                    subs_button = gr.Button(
                        lg_conf["button_subs"],
                        variant="primary",
                        visible=False,
                    )
                    subs_edit_space = gr.Textbox(
                        visible=False,
                        lines=10,
                        label=lg_conf["editor_sub_label"],
                        info=lg_conf["editor_sub_info"],
                        placeholder=lg_conf["editor_sub_ph"],
                    )
                    edit_sub_check.change(
                        visible_component_subs,
                        [edit_sub_check],
                        [subs_button, subs_edit_space],
                    )

                    with gr.Row():
                        video_button = gr.Button(
                            lg_conf["button_translate"],
                            variant="primary",
                        )
                    with gr.Row():
                        output_video_player = gr.Video(label="Output Video (.mp4)", interactive=False)
                    
                    with gr.Row():
                        video_output = gr.File(
                            label=lg_conf["output_result_label"],
                            file_count="multiple",
                            interactive=False,

                        )  # gr.Video()

                    gr.HTML("<hr></h2>")

                    if (
                        os.getenv("YOUR_HF_TOKEN") is None
                        or os.getenv("YOUR_HF_TOKEN") == ""
                    ):
                        HFKEY = gr.Textbox(
                            visible=True,
                            label="HF Token",
                            info=lg_conf["ht_token_info"],
                            placeholder=lg_conf["ht_token_ph"],
                        )
                    else:
                        HFKEY = gr.Textbox(
                            visible=False,
                            label="HF Token",
                            info=lg_conf["ht_token_info"],
                            placeholder=lg_conf["ht_token_ph"],
                        )

                    gr.Examples(
                        examples=[
                            [
                                ["./assets/Video_main.mp4"],
                                "",
                                "",
                                "",
                                False,
                                whisper_model_default,
                                4,
                                com_t_default,
                                "Spanish (es)",
                                "English (en)",
                                1,
                                2,
                                "en-CA-ClaraNeural-Female",
                                "en-AU-WilliamNeural-Male",
                            ],
                        ],  # no update
                        fn=SoniTr.batch_multilingual_media_conversion,
                        inputs=[
                            video_input,
                            blink_input,
                            directory_input,
                            HFKEY,
                            PREVIEW,
                            WHISPER_MODEL_SIZE,
                            batch_size,
                            compute_type,
                            SOURCE_LANGUAGE,
                            TRANSLATE_AUDIO_TO,
                            min_speakers,
                            max_speakers,
                            tts_voice00,
                            tts_voice01,
                        ],
                        outputs=[video_output],
                        cache_examples=False,
                    )

        with gr.Tab(lg_conf["tab_docs"]):
            with gr.Column():
                with gr.Accordion("Docs", open=True):
                    with gr.Column(variant="compact"):
                        with gr.Column():
                            input_doc_type = gr.Dropdown(
                                [
                                    "WRITE TEXT",
                                    "SUBMIT DOCUMENT",
                                    "Find Document Path",
                                ],
                                value="SUBMIT DOCUMENT",
                                label=lg_conf["docs_input_label"],
                                info=lg_conf["docs_input_info"],
                            )

                            def swap_visibility(data_type):
                                if data_type == "WRITE TEXT":
                                    return (
                                        gr.update(visible=True, value=""),
                                        gr.update(visible=False, value=None),
                                        gr.update(visible=False, value=""),
                                    )
                                elif data_type == "SUBMIT DOCUMENT":
                                    return (
                                        gr.update(visible=False, value=""),
                                        gr.update(visible=True, value=None),
                                        gr.update(visible=False, value=""),
                                    )
                                elif data_type == "Find Document Path":
                                    return (
                                        gr.update(visible=False, value=""),
                                        gr.update(visible=False, value=None),
                                        gr.update(visible=True, value=""),
                                    )

                            text_docs = gr.Textbox(
                                label="Text",
                                value="This is an example",
                                info="Write a text",
                                placeholder="...",
                                lines=5,
                                visible=False,
                            )
                            input_docs = gr.File(
                                label="Document", visible=True
                            )
                            directory_input_docs = gr.Textbox(
                                visible=False,
                                label="Document Path",
                                info="Example: /home/my_doc.pdf",
                                placeholder="Path goes here...",
                            )
                            input_doc_type.change(
                                fn=swap_visibility,
                                inputs=input_doc_type,
                                outputs=[
                                    text_docs,
                                    input_docs,
                                    directory_input_docs,
                                ],
                            )

                            gr.HTML()

                            tts_documents = gr.Dropdown(
                                list(
                                    filter(
                                        lambda x: x != "_XTTS_/AUTOMATIC.wav",
                                        SoniTr.tts_info.tts_list(),
                                    )
                                ),
                                value="en-US-EmmaMultilingualNeural-Female",
                                label="TTS",
                                visible=True,
                                interactive=True,
                            )

                            gr.HTML()

                            docs_SOURCE_LANGUAGE = gr.Dropdown(
                                LANGUAGES_LIST[1:],
                                value="English (en)",
                                label=lg_conf["sl_label"],
                                info=lg_conf["docs_source_info"],
                            )
                            docs_TRANSLATE_TO = gr.Dropdown(
                                LANGUAGES_LIST[1:],
                                value="English (en)",
                                label=lg_conf["tat_label"],
                                info=lg_conf["tat_info"],
                            )

                            with gr.Column():
                                with gr.Accordion(
                                    lg_conf["extra_setting"], open=False
                                ):
                                    docs_translate_process_dropdown = gr.Dropdown(
                                        DOCS_TRANSLATION_PROCESS_OPTIONS,
                                        value=DOCS_TRANSLATION_PROCESS_OPTIONS[
                                            0
                                        ],
                                        label="Translation process",
                                    )

                                    gr.HTML("<hr></h2>")

                                    docs_output_type = gr.Dropdown(
                                        DOCS_OUTPUT_TYPE_OPTIONS,
                                        value=DOCS_OUTPUT_TYPE_OPTIONS[2],
                                        label="Output type",
                                    )
                                    docs_OUTPUT_NAME = gr.Textbox(
                                        label="Final file name",
                                        value="",
                                        info=lg_conf["out_name_info"],
                                    )
                                    docs_chunk_size = gr.Number(
                                        label=lg_conf["chunk_size_label"],
                                        value=0,
                                        visible=True,
                                        interactive=True,
                                        info=lg_conf["chunk_size_info"],
                                    )
                                    gr.HTML("<hr></h2>")
                                    start_page_gui = gr.Number(
                                        step=1,
                                        value=1,
                                        minimum=1,
                                        maximum=99999,
                                        label="Start page",
                                    )
                                    end_page_gui = gr.Number(
                                        step=1,
                                        value=99999,
                                        minimum=1,
                                        maximum=99999,
                                        label="End page",
                                    )
                                    gr.HTML("<hr>Videobook config</h2>")
                                    videobook_width_gui = gr.Number(
                                        step=1,
                                        value=1280,
                                        minimum=100,
                                        maximum=4096,
                                        label="Width",
                                    )
                                    videobook_height_gui = gr.Number(
                                        step=1,
                                        value=720,
                                        minimum=100,
                                        maximum=4096,
                                        label="Height",
                                    )
                                    videobook_bcolor_gui = gr.Dropdown(
                                        BORDER_COLORS,
                                        value=BORDER_COLORS[0],
                                        label="Border color",
                                    )
                                    docs_dummy_check = gr.Checkbox(
                                        True, visible=False
                                    )

                            with gr.Row():
                                docs_button = gr.Button(
                                    lg_conf["docs_button"],
                                    variant="primary",
                                )
                            with gr.Row():
                                docs_output = gr.File(
                                    label="Result",
                                    interactive=False,
                                )

        with gr.Tab("Custom voice R.V.C. (Optional)"):

            with gr.Column():
                with gr.Accordion("Get the R.V.C. Models", open=True):
                    url_links = gr.Textbox(
                        label="URLs",
                        value="",
                        info=lg_conf["cv_url_info"],
                        placeholder="urls here...",
                        lines=1,
                    )
                    download_finish = gr.HTML()
                    download_button = gr.Button("DOWNLOAD MODELS")

                    def update_models():
                        models_path, index_path = upload_model_list()

                        dict_models = {
                            f"fmodel{i:02d}": gr.update(
                                choices=models_path
                            )
                            for i in range(MAX_TTS+1)
                        }
                        dict_index = {
                            f"findex{i:02d}": gr.update(
                                choices=index_path, value=None
                            )
                            for i in range(MAX_TTS+1)
                        }
                        dict_changes = {**dict_models, **dict_index}
                        return [value for value in dict_changes.values()]

            with gr.Column():
                with gr.Accordion(lg_conf["replace_title"], open=False):
                    with gr.Column(variant="compact"):
                        with gr.Column():
                            gr.Markdown(lg_conf["sec1_title"])
                            enable_custom_voice = gr.Checkbox(
                                False,
                                label="ENABLE",
                                info=lg_conf["enable_replace"]
                            )
                            workers_custom_voice = gr.Number(
                                step=1,
                                value=1,
                                minimum=1,
                                maximum=50,
                                label="workers",
                                visible=False,
                            )

                            gr.Markdown(lg_conf["sec2_title"])
                            gr.Markdown(lg_conf["sec2_subtitle"])

                            PITCH_ALGO_OPT = [
                                "pm",
                                "harvest",
                                "crepe",
                                "rmvpe",
                                "rmvpe+",
                            ]

                            def model_conf():
                                return gr.Dropdown(
                                    models_path,
                                    # value="",
                                    label="Model",
                                    visible=True,
                                    interactive=True,
                                )

                            def pitch_algo_conf():
                                return gr.Dropdown(
                                    PITCH_ALGO_OPT,
                                    value=PITCH_ALGO_OPT[3],
                                    label="Pitch algorithm",
                                    visible=True,
                                    interactive=True,
                                )

                            def pitch_lvl_conf():
                                return gr.Slider(
                                    label="Pitch level",
                                    minimum=-24,
                                    maximum=24,
                                    step=1,
                                    value=0,
                                    visible=True,
                                    interactive=True,
                                )

                            def index_conf():
                                return gr.Dropdown(
                                    index_path,
                                    value=None,
                                    label="Index",
                                    visible=True,
                                    interactive=True,
                                )

                            def index_inf_conf():
                                return gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label="Index influence",
                                    value=0.75,
                                )

                            def respiration_filter_conf():
                                return gr.Slider(
                                    minimum=0,
                                    maximum=7,
                                    label="Respiration median filtering",
                                    value=3,
                                    step=1,
                                    interactive=True,
                                )

                            def envelope_ratio_conf():
                                return gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label="Envelope ratio",
                                    value=0.25,
                                    interactive=True,
                                )

                            def consonant_protec_conf():
                                return gr.Slider(
                                    minimum=0,
                                    maximum=0.5,
                                    label="Consonant breath protection",
                                    value=0.5,
                                    interactive=True,
                                )

                            def button_conf(tts_name):
                                return gr.Button(
                                    lg_conf["cv_button_apply"]+" "+tts_name,
                                    variant="primary",
                                )

                            TTS_TABS = [
                                'TTS Speaker {:02d}'.format(i) for i in range(1, MAX_TTS+1)
                            ]

                            CV_SUBTITLES = [
                                lg_conf["cv_tts1"],
                                lg_conf["cv_tts2"],
                                lg_conf["cv_tts3"],
                                lg_conf["cv_tts4"],
                                lg_conf["cv_tts5"],
                                lg_conf["cv_tts6"],
                                lg_conf["cv_tts7"],
                                lg_conf["cv_tts8"],
                                lg_conf["cv_tts9"],
                                lg_conf["cv_tts10"],
                                lg_conf["cv_tts11"],
                                lg_conf["cv_tts12"],
                            ]

                            configs_storage = []

                            for i in range(MAX_TTS):  # Loop from 00 to 11
                                with gr.Accordion(CV_SUBTITLES[i], open=False):
                                    gr.Markdown(TTS_TABS[i])
                                    with gr.Column():
                                        tag_gui = gr.Textbox(
                                            value=TTS_TABS[i], visible=False
                                        )
                                        model_gui = model_conf()
                                        pitch_algo_gui = pitch_algo_conf()
                                        pitch_lvl_gui = pitch_lvl_conf()
                                        index_gui = index_conf()
                                        index_inf_gui = index_inf_conf()
                                        rmf_gui = respiration_filter_conf()
                                        er_gui = envelope_ratio_conf()
                                        cbp_gui = consonant_protec_conf()

                                        with gr.Row(variant="compact"):
                                            button_config = button_conf(
                                                TTS_TABS[i]
                                            )

                                            confirm_conf = gr.HTML()

                                        button_config.click(
                                            SoniTr.vci.apply_conf,
                                            inputs=[
                                                tag_gui,
                                                model_gui,
                                                pitch_algo_gui,
                                                pitch_lvl_gui,
                                                index_gui,
                                                index_inf_gui,
                                                rmf_gui,
                                                er_gui,
                                                cbp_gui,
                                            ],
                                            outputs=[confirm_conf],
                                        )

                                        configs_storage.append({
                                            "tag": tag_gui,
                                            "model": model_gui,
                                            "index": index_gui,
                                        })

                with gr.Column():
                    with gr.Accordion("Test R.V.C.", open=False):
                        with gr.Row(variant="compact"):
                            text_test = gr.Textbox(
                                label="Text",
                                value="This is an example",
                                info="write a text",
                                placeholder="...",
                                lines=5,
                            )
                            with gr.Column():
                                tts_test = gr.Dropdown(
                                    sorted(SoniTr.tts_info.list_edge),
                                    value="en-GB-ThomasNeural-Male",
                                    label="TTS",
                                    visible=True,
                                    interactive=True,
                                )
                                model_test = model_conf()
                                index_test = index_conf()
                                pitch_test = pitch_lvl_conf()
                                pitch_alg_test = pitch_algo_conf()
                        with gr.Row(variant="compact"):
                            button_test = gr.Button("Test audio")

                        with gr.Column():
                            with gr.Row():
                                original_ttsvoice = gr.Audio()
                                ttsvoice = gr.Audio()

                            button_test.click(
                                SoniTr.vci.make_test,
                                inputs=[
                                    text_test,
                                    tts_test,
                                    model_test,
                                    index_test,
                                    pitch_test,
                                    pitch_alg_test,
                                ],
                                outputs=[ttsvoice, original_ttsvoice],
                            )

                    download_button.click(
                        download_list,
                        [url_links],
                        [download_finish],
                        queue=False
                    ).then(
                        update_models,
                        [],
                        [
                            elem["model"] for elem in configs_storage
                        ] + [model_test] + [
                            elem["index"] for elem in configs_storage
                        ] + [index_test],
                    )

        with gr.Tab(lg_conf["tab_help"]):
            gr.Markdown(lg_conf["tutorial"])
            gr.Markdown(news)

            def play_sound_alert(play_sound):

                if not play_sound:
                    return None

                # silent_sound = "assets/empty_audio.mp3"
                sound_alert = "assets/sound_alert.mp3"

                time.sleep(0.25)
                # yield silent_sound
                yield None

                time.sleep(0.25)
                yield sound_alert

            sound_alert_notification = gr.Audio(
                value=None,
                type="filepath",
                format="mp3",
                autoplay=True,
                visible=False,
            )

        if logs_in_gui:
            logger.info("Logs in gui need public url")

            class Logger:
                def __init__(self, filename):
                    self.terminal = sys.stdout
                    self.log = open(filename, "w")

                def write(self, message):
                    self.terminal.write(message)
                    self.log.write(message)

                def flush(self):
                    self.terminal.flush()
                    self.log.flush()

                def isatty(self):
                    return False

            sys.stdout = Logger("output.log")

            def read_logs():
                sys.stdout.flush()
                with open("output.log", "r") as f:
                    return f.read()

            with gr.Accordion("Logs", open=False):
                logs = gr.Textbox(label=">>>")
                app.load(read_logs, None, logs, every=1)

        if SoniTr.tts_info.xtts_enabled:
            # Update tts list
            def update_tts_list():
                update_dict = {
                    f"tts_voice{i:02d}": gr.update(choices=SoniTr.tts_info.tts_list())
                    for i in range(MAX_TTS)
                }
                update_dict["tts_documents"] = gr.update(
                    choices=list(
                        filter(
                            lambda x: x != "_XTTS_/AUTOMATIC.wav",
                            SoniTr.tts_info.tts_list(),
                        )
                    )
                )
                return [value for value in update_dict.values()]

            create_xtts_wav.click(
                create_wav_file_vc,
                inputs=[
                    wav_speaker_name,
                    wav_speaker_file,
                    wav_speaker_start,
                    wav_speaker_end,
                    wav_speaker_dir,
                    wav_speaker_dereverb,
                ],
                outputs=[wav_speaker_output],
            ).then(
                update_tts_list,
                None,
                [
                    tts_voice00,
                    tts_voice01,
                    tts_voice02,
                    tts_voice03,
                    tts_voice04,
                    tts_voice05,
                    tts_voice06,
                    tts_voice07,
                    tts_voice08,
                    tts_voice09,
                    tts_voice10,
                    tts_voice11,
                    tts_documents,
                ],
            )

        # Run translate text
        subs_button.click(
            SoniTr.batch_multilingual_media_conversion,
            inputs=[
                video_input,
                blink_input,
                directory_input,
                HFKEY,
                PREVIEW,
                WHISPER_MODEL_SIZE,
                batch_size,
                compute_type,
                SOURCE_LANGUAGE,
                TRANSLATE_AUDIO_TO,
                min_speakers,
                max_speakers,
                tts_voice00,
                tts_voice01,
                tts_voice02,
                tts_voice03,
                tts_voice04,
                tts_voice05,
                tts_voice06,
                tts_voice07,
                tts_voice08,
                tts_voice09,
                tts_voice10,
                tts_voice11,
                VIDEO_OUTPUT_NAME,
                AUDIO_MIX,
                audio_accelerate,
                acceleration_rate_regulation_gui,
                volume_original_mix,
                volume_translated_mix,
                sub_type_output,
                edit_sub_check,  # TRUE BY DEFAULT
                dummy_false_check,  # dummy false
                subs_edit_space,
                avoid_overlap_gui,
                vocal_refinement_gui,
                literalize_numbers_gui,
                segment_duration_limit_gui,
                diarization_process_dropdown,
                translate_process_dropdown,
                input_srt,
                main_output_type,
                main_voiceless_track,
                voice_imitation_gui,
                voice_imitation_max_segments_gui,
                voice_imitation_vocals_dereverb_gui,
                voice_imitation_remove_previous_gui,
                voice_imitation_method_gui,
                wav_speaker_dereverb,
                text_segmentation_scale_gui,
                divide_text_segments_by_gui,
                soft_subtitles_to_video_gui,
                burn_subtitles_to_video_gui,
                enable_cache_gui,
                enable_custom_voice,
                workers_custom_voice,
                is_gui_dummy_check,
            ],
            outputs=subs_edit_space,
        ).then(
            play_sound_alert, [play_sound_gui], [sound_alert_notification]
        )

        # Run translate tts and complete
        video_button.click(
            SoniTr.batch_multilingual_media_conversion,
            inputs=[
                video_input,
                blink_input,
                directory_input,
                HFKEY,
                PREVIEW,
                WHISPER_MODEL_SIZE,
                batch_size,
                compute_type,
                SOURCE_LANGUAGE,
                TRANSLATE_AUDIO_TO,
                min_speakers,
                max_speakers,
                tts_voice00,
                tts_voice01,
                tts_voice02,
                tts_voice03,
                tts_voice04,
                tts_voice05,
                tts_voice06,
                tts_voice07,
                tts_voice08,
                tts_voice09,
                tts_voice10,
                tts_voice11,
                VIDEO_OUTPUT_NAME,
                AUDIO_MIX,
                audio_accelerate,
                acceleration_rate_regulation_gui,
                volume_original_mix,
                volume_translated_mix,
                sub_type_output,
                dummy_false_check,
                edit_sub_check,
                subs_edit_space,
                avoid_overlap_gui,
                vocal_refinement_gui,
                literalize_numbers_gui,
                segment_duration_limit_gui,
                diarization_process_dropdown,
                translate_process_dropdown,
                input_srt,
                main_output_type,
                main_voiceless_track,
                voice_imitation_gui,
                voice_imitation_max_segments_gui,
                voice_imitation_vocals_dereverb_gui,
                voice_imitation_remove_previous_gui,
                voice_imitation_method_gui,
                wav_speaker_dereverb,
                text_segmentation_scale_gui,
                divide_text_segments_by_gui,
                soft_subtitles_to_video_gui,
                burn_subtitles_to_video_gui,
                enable_cache_gui,
                enable_custom_voice,
                workers_custom_voice,
                is_gui_dummy_check,
            ],
            outputs=[video_output, output_video_player], 
            trigger_mode="multiple",
        ).then(
            play_sound_alert, [play_sound_gui], [sound_alert_notification]
        )

        # Run docs process
        docs_button.click(
            SoniTr.multilingual_docs_conversion,
            inputs=[
                text_docs,
                input_docs,
                directory_input_docs,
                docs_SOURCE_LANGUAGE,
                docs_TRANSLATE_TO,
                tts_documents,
                docs_OUTPUT_NAME,
                docs_translate_process_dropdown,
                docs_output_type,
                docs_chunk_size,
                enable_custom_voice,
                workers_custom_voice,
                start_page_gui,
                end_page_gui,
                videobook_width_gui,
                videobook_height_gui,
                videobook_bcolor_gui,
                docs_dummy_check,
            ],
            outputs=docs_output,
            trigger_mode="multiple",
        ).then(
            play_sound_alert, [play_sound_gui], [sound_alert_notification]
        )

    return app


def get_language_config(language_data, language=None, base_key="english"):
    base_lang = language_data.get(base_key)

    if language not in language_data:
        logger.error(
            f"Language {language} not found, defaulting to {base_key}"
        )
        return base_lang

    lg_conf = language_data.get(language, {})
    lg_conf.update((k, v) for k, v in base_lang.items() if k not in lg_conf)

    return lg_conf


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="Taithrah/Minimal",
        help=(
            "Specify the theme; find themes in "
            "https://huggingface.co/spaces/gradio/theme-gallery;"
            " Example: --theme aliabid94/new-theme"
        ),
    )
    parser.add_argument(
        "--public_url",
        action="store_true",
        default=False,
        help="Enable public link",
    )
    parser.add_argument(
        "--logs_in_gui",
        action="store_true",
        default=False,
        help="Displays the operations performed in Logs",
    )
    parser.add_argument(
        "--verbosity_level",
        type=str,
        default="info",
        help=(
            "Set logger verbosity level: "
            "debug, info, warning, error, or critical"
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        help=" Select the language of the interface: english, spanish",
    )
    parser.add_argument(
        "--cpu_mode",
        action="store_true",
        default=False,
        help="Enable CPU mode to run the program without utilizing GPU acceleration.",
    )
    return parser


if __name__ == "__main__":

    parser = create_parser()

    args = parser.parse_args()
    # Simulating command-line arguments
    # args_list = "--theme aliabid94/new-theme --public_url".split()
    # args = parser.parse_args(args_list)

    set_logging_level(args.verbosity_level)

    for id_model in UVR_MODELS:
        download_manager(
            os.path.join(MDX_DOWNLOAD_LINK, id_model), mdxnet_models_dir
        )

    models_path, index_path = upload_model_list()

    SoniTr = SoniTranslate(cpu_mode=args.cpu_mode)

    lg_conf = get_language_config(language_data, language=args.language)

    app = create_gui(args.theme, logs_in_gui=args.logs_in_gui)

    app.queue()
    # import webbrowser
    # webbrowser.open("http://127.0.0.1:7860")    
    app.launch(
        max_threads=1,
        share=args.public_url,
        show_error=True,
        quiet=False,
        debug=(True if logger.isEnabledFor(logging.DEBUG) else False),
    )
    
