# Copyright (c) Alibaba, Inc. and its affiliates.
import gradio as gr
from inference.qwen_infer import qwen_infer, PROMPT_TEMPLATE
from inference.clip_infer import clip_infer
from inference.sdxl_infer import sdxl_infer, STYLE_TEMPLATE, GENERAL_STYLE
from inference.I2VGen_infer import i2v_infer, v2v_infer
from inference.music_infer import music_infer

def script_gen():
    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=5>剧本生成(Script Generation)</center>""")
        with gr.Box():
            with gr.Row():
                with gr.Column(scale=1):
                    theme = gr.Textbox(label='主题(Theme)', placeholder='请输入剧本主题，如“未来科幻片“\n(Please enter the theme of the script, e.g., science fiction film.)', lines=2)
                    background= gr.Textbox(label='背景(Background)', placeholder='请输入剧本背景，如“太空”\n(Please enter the script background, e.g., space.)', lines=2)
                    scenario = gr.Textbox(label='剧情要求(Plot)', placeholder='请输入剧情要求，如“充满想象力，跌宕起伏”\n(Please enter plot requirements, e.g., "imaginative, suspenseful ups and downs".)', lines=3)
                    language = gr.Radio(choices=['中文(Chinese)', '英文(English)'], label='语言(Language)', value='中文(Chinese)', interactive=True)
                    act = gr.Slider(minimum=1, maximum=6, value=3, step=1, interactive=True, label='剧本幕数(The number of scenes in the script)')
                    with gr.Row():
                        clear_script = gr.Button('清空(Clear)')
                        submit_script = gr.Button('生成剧本(Submit)')
                with gr.Column(scale=2):
                    script = gr.Textbox(label='剧本(Script) BY Qwen-7B-Chat', interactive=False, lines=15)
        
        with gr.Box():
            with gr.Accordion("图片生成微剧本(Image to story)", open=False):
                with gr.Row():
                    with gr.Column(scale=1): 
                        image = gr.Image(label='图片(Image)', type="filepath", interactive=True, height=200)
                        story_theme = gr.Textbox(label='主题(Theme)', placeholder='请输入剧本主题，如“爱情片“\n(Please enter the theme of the story, e.g., affectional film.)', lines=2)
                        with gr.Row():
                            clear_story = gr.Button('清空(Clear)')
                            submit_story = gr.Button('生成微剧本(Submit)')
                    with gr.Column(scale=2):
                        story = gr.Textbox(label='微剧本(Story) BY Qwen-7B-Chat', interactive=False, lines=8)


        def qwen_script(theme, background, act, scenario, language):
            inputs = PROMPT_TEMPLATE['script'].format(theme=theme, background=background, act=act, scenario=scenario, language=language)
            script = qwen_infer(inputs=inputs)
            return script
        
        def qwen_story(story_theme, image):
            picture = clip_infer(image)
            inputs = PROMPT_TEMPLATE['story'].format(story_theme=story_theme, picture=picture)
            story = qwen_infer(input=inputs)
            return story

        submit_script.click(qwen_script,
                    inputs=[theme, background, act, scenario, language], 
                    outputs=[script])
        clear_script.click(lambda: [None, None, 3, None, None], 
                    inputs=[], 
                    outputs=[theme, background, act, scenario, script])
        
        submit_story.click(qwen_story,
                    inputs=[story_theme, image], 
                    outputs=[story])
        clear_story.click(lambda: [None, None, None], 
                    inputs=[], 
                    outputs=[story_theme, image, story])
    return demo, script

def production_still_gen():
    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=5>剧照生成(Movie still Generation)</center>""")
        with gr.Box():
            gr.Markdown("""<left><font size=3>Step 1: 填入一幕剧本，然后点击“生成”，可以得到对应剧本的剧照场景描述和文生图提示词。</left>""")
            gr.Markdown("""<left><font size=3>Step 1: Enter a script scene, then click "Submit" to obtain the corresponding movie still scene description and the prompt.</left>""")
            with gr.Row():
                with gr.Column(scale=1):
                    script = gr.Textbox(label='剧本(Script)', placeholder='请输入剧本中的一幕\n(Please enter a scene from the script.)', lines=8)
                    language = gr.Radio(choices=['中文(Chinese)', '英文(English)'], label='语言(Language)', value='中文(Chinese)', interactive=True)
                    with gr.Row():
                        clear_prompt = gr.Button('清空(Clear)')
                        submit_prompt = gr.Button('生成(Submit)')
                with gr.Column(scale=2):
                    still_description = gr.Textbox(label='剧照描述(Movie still description) BY Qwen-7B-Chat', lines=5, interactive=False)
                    SD_prompt = gr.Textbox(label='提示词(Prompt) BY Qwen-7B-Chat', lines=5, interactive=False)

            def qwen_still(script, language):
                inputs = PROMPT_TEMPLATE['still'].format(script=script, language=language)
                still_description = qwen_infer(inputs=inputs)
                return still_description
            
            def qwen_sd_prompt(still_description):
                inputs = PROMPT_TEMPLATE['SD'].format(still_description=still_description)
                SD_prompt = qwen_infer(inputs=inputs)
                return SD_prompt

            submit_prompt.click(qwen_still, 
                                inputs=[script, language], 
                                outputs=[still_description]).then(qwen_sd_prompt, inputs=[still_description], outputs=[SD_prompt])
            clear_prompt.click(lambda: [None, None, None], 
                        inputs=[], 
                        outputs=[script, still_description, SD_prompt])
        
        with gr.Box():
            gr.Markdown("""<left><font size=3>Step 2: 复制上一步中得到的提示词，选择合适的风格和参数，点击“生成”，得到剧照。</left>""")
            gr.Markdown("""<left><font size=3>Step 2: Copy the prompt obtained in the Step 1, select appropriate styles and parameters, click "Submit" to obtain a movie still.</left>""")
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(label='提示词(Prompts)',lines=3)
                    negative_prompt = gr.Textbox(label='负向提示词(Negative Prompts)',lines=3)
                    with gr.Row():
                        height = gr.Slider(512, 1024, 1024, step=128, label='高度(Height)', interactive=True)
                        width = gr.Slider(512, 1024, 1024, step=128, label='宽度(Width)', interactive=True)
                    with gr.Row():
                        scale = gr.Slider(1, 15, 10, step=.25, label='引导系数(CFG scale)', interactive=True)
                        steps = gr.Slider(25, maximum=100, value=50, step=5, label='迭代步数(Steps)', interactive=True)
                    seed = gr.Slider(minimum=1, step=1, maximum=999999999999999999, randomize=True, label='随机数种子(Seed)', interactive=True)

                with gr.Column(scale=3):
                    output_image = gr.Image(label='剧照(Movie still) BY SDXL-1.0', interactive=False, height=400)
                    with gr.Row():
                        clear = gr.Button('清空(Clear)')
                        submit = gr.Button('生成(Submit)')

                submit.click(sdxl_infer, inputs=[prompt, negative_prompt, height, width, scale, steps, seed], outputs=output_image)
                clear.click(lambda: [None, None, 1024, 1024, 10, 50, None], inputs=[], outputs=[prompt, negative_prompt, height, width, scale, steps, output_image])

            with gr.Accordion("提示词助手(Prompt assistant)", open=False):
                gr.Markdown("""<left><font size=2>您可以在此定制风格提示词，复制到提示词框中。固定风格提示词可以一定程度上固定剧照的风格。(You can customize your own style prompts here and copy them into the prompt textbox. Using consistent style prompts can help maintain a unified style of the movie stills to some degree.)</left>""")
                workbench = gr.Textbox(label="提示词工作台(Prompt workbench)", interactive=True, lines=2)
                general_style = gr.Radio(list(GENERAL_STYLE.keys()), value='无(None)', label='大致风格(General styles)', interactive=True)
                with gr.Row():
                    with gr.Column():
                        art = gr.Dropdown(STYLE_TEMPLATE['art'], multiselect=True, label="艺术风格(Art)")
                        atmosphere = gr.Dropdown(STYLE_TEMPLATE['atmosphere'], multiselect=True, label="场景氛围(Atmosphere)")
                        illustration_style = gr.Dropdown(STYLE_TEMPLATE['illustration style'], multiselect=True, label="插画风格(Illustration style)")
                    with gr.Column():
                        theme = gr.Dropdown(STYLE_TEMPLATE['theme'], multiselect=True, label="主题(Theme)")
                        image_quality = gr.Dropdown(STYLE_TEMPLATE['image quality'], multiselect=True, label="画质(Image quality)")
                        lighting = gr.Dropdown(STYLE_TEMPLATE['lighting'], multiselect=True, label="光照(Lighting)")
                    with gr.Column():
                        lens_style = gr.Dropdown(STYLE_TEMPLATE['lens style'], multiselect=True, label="相机镜头(Lens style)")
                        character_shot = gr.Dropdown(STYLE_TEMPLATE['character shot'], multiselect=True, label="人物镜头(Character shot)")
                        view = gr.Dropdown(STYLE_TEMPLATE['view'], multiselect=True, label="视角(View)")

                submit_style = gr.Button("提交至工作台(Submit to workbench)")

                STYLE_NAME = [general_style, art, atmosphere, illustration_style, theme, image_quality, lighting, lens_style, character_shot, view]
                def update_workbench(*styles):
                    style_prompt = GENERAL_STYLE[styles[0]]
                    style_list = []
                    for style in styles[1:]:
                        for word in style:
                            style_list.append(word)
                    style_prompt += ", ".join(style_list)
                    return style_prompt
                
                submit_style.click(fn=update_workbench,
                                    inputs=STYLE_NAME,
                                    outputs=[workbench],
                                    queue=False)
            
    return demo

def video_gen():
    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=5>视频生成(Video Generation)</center>""")
        with gr.Box():
            gr.Markdown("""<left><font size=3>Step 1: 上传剧照（建议图片比例为1:1），然后点击“生成”，得到满意的视频后进行下一步。</left>""")
            gr.Markdown("""<left><font size=3>Step 1: Upload a movie still (it is recommended that the image ratio is 1:1), then click "Submit" to get a satisfactory video before moving to the Step 2.</left>""")
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(label="剧照(Movie still)", type="filepath", interactive=True, height=300)
                    with gr.Row():
                        clear_image = gr.Button("清空(Clear)")
                        submit_image = gr.Button("生成视频(Submit)")
                with gr.Column():
                    video_out_1 = gr.Video(label='视频(Video) BY I2VGen-XL', interactive=False, height=300)
        with gr.Box():
            gr.Markdown("""<left><font size=3>Step 2: 补充对视频内容的英文文本描述，然后点击“生成高分辨率视频”。</left>""")
            gr.Markdown("""<left><font size=3>Step 2: Add the English text description of the video you want to generate, then click "Submit".</left>""")
            with gr.Row():
                with gr.Column():
                    text_in = gr.Textbox(label="视频描述(Video description)", placeholder='请输入对视频场景的英文描述\n(Please enter a description of the video scene.)', lines=8)
                    with gr.Row():
                        clear_video = gr.Button("清空(Clear)")
                        submit_video = gr.Button("生成高分辨率视频(Submit)")
                with gr.Column():
                    video_out_2 = gr.Video(label='高分辨率视频(High-resolutions video) BY MS-Vid2Vid-XL', interactive=False, height=300)
    
        submit_image.click(i2v_infer, inputs=[image_in], outputs=[video_out_1])
        submit_video.click(v2v_infer, inputs=[video_out_1, text_in], outputs=[video_out_2])
        clear_image.click(lambda: [None, None], inputs=[], outputs=[image_in, video_out_1])
        clear_video.click(lambda: [None, None], inputs=[], outputs=[text_in, video_out_2])

    return demo


def music_gen():
    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=5>音乐生成(Music Generation)</center>""")
        with gr.Row():
            with gr.Column():
                description = gr.Text(label="音乐描述(Music description)", interactive=True, lines=10)
                duration = gr.Slider(minimum=1, maximum=30, value=10, label="生成时长(Duration)", interactive=True)
                # model_id = gr.Radio(["small"], label="模型(Model)", value="small", interactive=True)
                with gr.Row():
                    clear = gr.Button("清空(Clear)")
                    submit = gr.Button("生成(Submit)")
                # with gr.Accordion("更多设置(Advanced Options)", open=False):
                #     with gr.Row():
                #         topk = gr.Number(label="Top-k", value=250, interactive=True)
                #         topp = gr.Number(label="Top-p", value=0, interactive=True)
                #         temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                #         cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
            with gr.Column():
                output = gr.Video(label="Music BY MusicGen", interactive=False)

            submit.click(music_infer, inputs=[description, duration], outputs=[output])
            clear.click(lambda: ["small", None, 10, None], inputs=[], outputs=[description, duration, output])

    return demo



with gr.Blocks() as demo:
    gr.Markdown("""<center><font size=8>MotionAgent</center>""")
    with gr.Tabs():
        with gr.TabItem('剧本生成(Script Generation)'):
            script_gen()
        with gr.TabItem('剧照生成(Movie still Generation)'):
            production_still_gen()
        with gr.TabItem('视频生成(Video Generation)'):
            video_gen()
        with gr.TabItem('音乐生成(Music Generation)'):
            music_gen()

demo.queue(status_update_rate=1).launch(share=True)
