import cv2
import torch
import gradio as gr
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline

GENERAL_STYLE={
    "无(None)": "",
    "增强(Enhance)": "breathtaking, award-winning, professional, highly detailed, ",
    "动漫(Anime)": "anime artwork, anime style, key visual, vibrant, studio anime, highly detailed, ",
    "摄影(Photographic)": "cinematic photo, 35mm photograph, film, bokeh, professional, 4k, highly detailed, ",
    "数字艺术(Digital Art)": "concept art, digital artwork, illustrative, painterly, matte painting, highly detailed, ",
    "漫画书(Comic Book)": "comic, graphic illustration, comic art, graphic novel art, vibrant, highly detailed, ",
    "奇幻艺术(Fantasy Art)": "ethereal fantasy concept art of, magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy, ",
    "模拟胶片(Analog Film)": "analog film photo, faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, ",
    "霓虹朋克(Neon Punk)": "neonpunk style, cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional, ",
    "等距风格(Isometric)": "isometric style, vibrant, beautiful, crisp, detailed, ultra detailed, intricate, ",
    "折纸工艺(Origami)": "origami style, paper art, pleated paper, folded, origami art, pleats, cut and fold, centered composition, ",
    "线描艺术(Line Art)": "line art drawing, professional, sleek, modern, minimalist, graphic, line art, vector graphics, ",
    "工艺粘土(Craft Clay)": "play-doh style, sculpture, clay art, centered composition, Claymation, ",
    "影视风格(Cinematic)": "cinematic film still, shallow depth of field, vignette, highly detailed, high budget Hollywood movie, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy, ",
    "3D模型(3D Model)": "professional 3d model, octane render, highly detailed, volumetric, dramatic lighting, ",
    "像素艺术(Pixel Art)": "pixel-art, low-res, blocky, pixel art style, 8-bit graphics, ",
    }


STYLE_TEMPLATE = {
    "art": ["Tradition Chinese Ink Painting style","Photograph","digital art","IKEA guide","comic book","baroque art style","impressionism art style","portrait art style","instruction manual","Newspaper","3D rendering","Photorealism ","Magazine","Mosaic","art nouveau","fresco art style ","pixel art style","surrealism art style","movie poster"],
    "atmosphere": ["Dark Atmosphere","Reflective Atmosphere","Enchanting Atmosphere","Mystical Atmosphere","Whimsical Atmosphere","Enigmatic Atmosphere","Ethereal Atmosphere","Tranquil Atmosphere","Relaxing Atmosphere","Blissful Atmosphere","Moody Atmosphere","Intense Atmosphere","Nostalgic Atmosphere","Industrial Atmosphere","Gothic Atmosphere","Light Atmosphere","Hazy Atmosphere","Dreamy Atmosphere","Playful Atmosphere","Mysterious Atmosphere","Mellow Atmosphere","Peaceful Atmosphere","Sophisticated Atmosphere","Sophisticated Atmosphere","Zen Atmosphere","Chill Atmosphere","Melancholic Atmosphere","Festive Atmosphere","Rustic Atmosphere","Romantic Atmosphere"],
    "illustration style": ["Caricature","Children’s Drawing","Doodle","Drawing","Figure Drawing","Hand-Drawn","Illuminated Manuscript","Illustration","Masterpiece","Sketch","Storybook Illustration","Whimsical Illustration","Blackboard","Chalk","Colored Pencil","Conte","Dry-Erase Marker","Fountain Pen Art","Graphite","Grease Pencil","Ink","Marker Art","Pencil Art","Wet-Erase Marker","1980s Airbrush Art","Ancient Roman Painting","Blacklight Paint","Canvas","Casein Paint","Chinese Painting","Color Field Painting","Dripping Paint","Egg Decorating","Faux Painting","Fine Art","Gond Painting","Graffiti","Hydro-Dipping","Japanese Painting","Madhubani Painting","Modern Art","Oil Paint","Paintwork","Phad Painting","Rock Art","Sandpainting","Speedpainting","Spray ","Still-Life","Tibetan Painting","Watercolor","Assembly Drawing","Cartographic","Crosshatch","Dot Art","Etch-A-Sketch Drawing","Graphic Novel","Hand-Written","Illustrated-Booklet","Line Art","Pointillism","Stipple","Visual Novel","Ballpoint Pen","Calligraphy","Charcoal Art","Conductive Ink","Crayon","Flexographic Ink","Gel Pen","Grease Pen","India Ink","Iron Gall Ink","Pastel Art","Viscosity Print","Whiteboard","Airbrush","Artwork","Brushwork","Caravaggio Painting","Cave Art","Coffee Paint","Detailed Painting","Easter Egg","Encaustic Painting","Fayum Portrait","Glass Paint","Gouache Paint","Hard Edge Painting","Impasto","Kalamkari Painting","Matte Painting","Mural","Painting","Paper-Marbling","Puffy Paint","Romanesque Painting","Scroll Painting","Splatter Paint","Stencil Graffiti","Street Art","Warli Painting","Wet Paint"],
    "theme": ["Hyper Real","Photorealism","Fantastic Realism","Classical Realism","Contemporary Realism","Surrealism","Non-Fiction","Imagined","Imagination","Fever-Dream","Daydreampunk","Weirdcore","Otherworldly","Lucid","Ethereality","Déjà vu","Abstraction","Fantasy","Dark Fantasy","Illusion","Nonsense","Intangible","Visual Exaggeration","Exaggeration","Retrowave","Vintage","Cyberpunk","Nanopunk","Rollerwave","Rusticcore","Pre-Historic","Prehistoricore","Atompunk","Wild West","Modernismo","Retro-Futurism","Future Funk","Extraterrestrial","Sci-fi","Psychic","Decopunk","Fairy Folk","Anime","Horror Anime","Manga","UwU","Photorealistic","Realism","Magic Realism","New Realism","Surreal","Unrealistic","Science Fiction","Imaginative","Dreamlike","Dreampunk","Dreamcore","Worldly","Unworldly","Wonderland","Ethereal","Anemoiacore","Abstract","Lyrical Abstraction","Ethereal Fantasy","Fantasy Map","Impossible","Immaterial","Visual Rhetoric","Exaggerated","Retro","Nostalgiacore","Antique","Postcyberpunk","Raypunk","Rustic","Rococopunk","Historic","Jurassic","Ice Age","Modern","Futuristic","Cassette Futurism","Afrofuturist","Invasion","Magic","Aetherpunk","Dragoncore","Mythpunk","Cartoon","Kawaii","Marvel Comics","Vampirella"],
    "image quality": ["high detail","high resolution","hyperrealism","HD","16K","hyper quality","trending on artstation","surrealism","8K"],
    "lens style": ["DSLR","360 panorama","telephoto lens","super wide angle","tilt-shift","depth of field (dof)","microscopic view","super resolution microscopy"],
    "character shot": ["full body","Detail Shot(ECU)","Face Shot (VCU)","Big Close-Up(BCU)","Close-Up(CU)","Chest Shot(MCU)","Waist Shot(WS)"],
    "view": ["first-person view","A bird's-eye view,aerial view","Top view","Bottom view","cinematic shot","extreme long shot","long shot","Mid shot","extreme close up","medium close up","close up","scenery shot","satellite view"],
    "lighting": ["studio lighting","film lighting","beautiful lighting","Soft illumination","dramatic lighting","rim lights","Back lighting","Split Lighting","mood lighting","Volumetric lighting","Rembrandt Lighting","bioluminescence","Crepuscular Ray","rays of shimmering light"],
} 


def sdxl_infer(prompt: str,
               negative_prompt: str,
               height: int = 1024,
               width: int = 1024,
               scale: float = 10,
               steps: int = 50,
               seed: int = 0):
    pipe = pipeline(task=Tasks.text_to_image_synthesis,
                    model='AI-ModelScope/stable-diffusion-xl-base-1.0',
                    use_safetensors=True,
                    model_revision='v1.0.0')

    if not prompt:
        raise gr.Error('提示词不能为空。(Please enter the prompts.)')
    generator = torch.Generator(device='cuda').manual_seed(seed)
    output = pipe({'text': prompt,
                   'negative_prompt': negative_prompt,
                   'num_inference_steps': steps,
                   'guidance_scale': scale,
                   'height': height,
                   'width': width,
                   'generator': generator})
    result = output['output_imgs'][0]

    image_path = './lora_result.png'
    cv2.imwrite(image_path, result)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    del pipe
    torch.cuda.empty_cache()
    return image
