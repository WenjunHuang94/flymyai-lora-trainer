import os
import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool, cpu_count
from functools import partial
import json
import re
from functools import partial

# 配置
FONT_PATH = ""  # <--- 这个全局变量我们不用，测试时会传入找到的字体
IMAGE_SIZE = 512
NUM_PROCESSES = min(35, cpu_count())  # 使用的进程数，最多16个

FONT_CACHE = {}


def get_font(font_path, size):
    key = (font_path, size)
    if key not in FONT_CACHE:
        # 增加一个对 "default" 的处理，以防 find_system_font 失败
        if font_path is None or font_path == "default":
            print("警告: 正在使用 PIL 默认位图字体，效果可能不佳。")
            FONT_CACHE[key] = ImageFont.load_default()
        else:
            try:
                FONT_CACHE[key] = ImageFont.truetype(font_path, size)
            except IOError:
                print(f"错误: 无法加载字体 {font_path}。将使用默认字体。")
                FONT_CACHE[key] = ImageFont.load_default()
    return FONT_CACHE[key]


BREAK_CHARS = r"_\-/\.\|\\"

CAMEL_RE = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')


def measure_width(draw, font, s: str) -> int:
    if not s:
        return 0
    # 增加 try-except 以兼容 load_default() 返回的位图字体
    try:
        x0, y0, x1, y1 = draw.textbbox((0, 0), s, font=font)
        return x1 - x0
    except Exception:
        # 兜底
        return draw.textlength(s, font=font)


def split_camel(s: str):
    # 在驼峰边界插入空格再切
    return CAMEL_RE.sub(' ', s).split(' ')


def smart_tokenize(s: str):
    """
    智能分词：
    1) 先按分隔符拆开，并把分隔符作为独立token保留（用于优先换行）
    2) 再对每个非空token做一次驼峰拆分
    """
    if not s:
        return []
    # 先按分隔符拆，保留分隔符
    parts = re.split(f'([{BREAK_CHARS}])', s)
    tokens = []
    for part in parts:
        if part == '':
            continue
        if len(part) == 1 and re.match(f'[{BREAK_CHARS}]', part):
            # ✅ 下划线替换为空格，不保留
            if part == '_':
                tokens.append(' ')
            else:
                tokens.append(part)
        else:
            for seg in split_camel(part):
                if seg != '':
                    tokens.append(seg)
    return tokens


def split_long_token_hard(draw, font, token, max_width):
    """
    硬切策略（仅在极端情况下使用）：
    - 仅当整个单词比 max_width 还宽时才拆
    - 普通单词不拆开（保持整体）
    """
    tmp_img = Image.new("RGB", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp_img)
    width = measure_width(tmp_draw, font, token)

    # ✅ 如果单词整体宽度小于 max_width，不拆
    if width <= max_width:
        return [token]

    # ✅ 若单词整体太长但至少能显示一个字符，则尝试逐字符切
    # （这种情况几乎只发生在超长英文单词或无空格字符串）
    try:
        avg_char_width = measure_width(tmp_draw, font, "W")
    except Exception:  # 兼容位图字体
        avg_char_width = 8

    if width < 1.5 * max_width or avg_char_width * len(token) < 1.5 * max_width:
        return [token]

    # ✅ 否则强制逐字符切（兜底）
    out, cur = [], ""
    for ch in token:
        if measure_width(tmp_draw, font, cur + ch) <= max_width or not cur:
            cur += ch
        else:
            out.append(cur)
            cur = ch
    if cur:
        out.append(cur)
    return out


def wrap_line(draw, font, text, max_width):
    """
    折行算法（优先软断点与驼峰）：
    - 先用 smart_tokenize 得到 token 流（含分隔符独立token）
    - 测试追加 token 后是否超宽；若超宽且 token 本身超宽，调用硬切
    - 分隔符 token 若导致超宽，将其放到下一行开头（避免悬挂到行末）
    """
    if text == "":
        return [""]
    # print(f"[WRAP] text='{text}', max_width={max_width}")
    tokens = smart_tokenize(text)
    lines = []
    cur = ""

    for tok in tokens:
        candidate = (cur + tok) if (
                    cur == "" or cur.endswith(" ") or len(tok) == 1 and re.match(f'[{BREAK_CHARS}]', tok)) else (
                    cur + " " + tok)
        w = measure_width(draw, font, candidate)

        if w <= max_width:
            cur = candidate
        else:
            # 先收当前行（若有）
            if cur != "":
                lines.append(cur)
                cur = ""

            # 当前 token 自己超宽 -> 硬切后逐段放入
            if measure_width(draw, font, tok) > max_width:
                # ✅ 若是普通英文单词，整块换行，不拆
                if re.match(r'^[A-Za-z0-9]+$', tok):
                    if cur:
                        lines.append(cur.strip())
                    cur = tok
                    continue
                for piece in split_long_token_hard(draw, font, tok, max_width):
                    if measure_width(draw, font, piece) <= max_width:
                        if cur == "":
                            cur = piece
                        else:
                            cand2 = (cur + " " + piece)
                            if measure_width(draw, font, cand2) <= max_width:
                                cur = cand2
                            else:
                                lines.append(cur)
                                cur = piece
                    else:
                        # 极端情况：字符本身超宽（几乎不会发生），强制成独行
                        if cur:
                            lines.append(cur)
                        lines.append(piece)
                        cur = ""
            else:
                # token 不超宽但与当前行合并超宽 -> 放到新行
                cur = tok

    if cur != "":
        lines.append(cur)

    # 去除行首尾多余空格（可选）
    lines = [re.sub(r'\s+', ' ', ln.strip()) for ln in lines]
    # print(f"[WRAP] lines={lines}")
    lines = [ln.strip() for ln in lines if ln.strip()]

    if not lines and text:  # 兜底，如果啥也没有但原始文本有
        return [text]
    if not lines:
        return [""]  # 确保至少返回一个空字符串行

    return lines


def layout_for_font_size(text, font_path, font_size, image_size, margin, line_gap):
    font = get_font(font_path, font_size)
    tmp = Image.new("RGB", (image_size, image_size))
    draw = ImageDraw.Draw(tmp)

    max_width = image_size - 2 * margin
    max_height = image_size - 2 * margin

    paragraphs = text.split("\n")
    lines = []
    for para in paragraphs:
        wrapped = wrap_line(draw, font, para, max_width)
        if not wrapped:
            wrapped = [""]
        lines.extend(wrapped)

    try:
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
    except Exception:
        # 兼容位图字体
        line_height = measure_width(draw, font, "A") + 4  # 估算

    total_height = 0
    for i in range(len(lines)):
        total_height += line_height
        if i != len(lines) - 1:
            total_height += line_gap

    fits = total_height <= max_height
    # print(f"[LAYOUT] font={font_size}, lines={len(lines)}, total_height={total_height}, "
    #       f"max_height={max_height}, fits={fits}, first_line='{lines[0] if lines else ''}'")
    return fits, lines, font, line_height, total_height


def render_text_with_adaptive_font(
        text, font_path, image_size, text_color, background_color, margin=10, line_gap=4
):
    left, right = 10, 200
    best = None

    while left <= right:
        mid = (left + right) // 2
        try:
            fits, lines, font, line_height, total_height = layout_for_font_size(
                text, font_path, mid, image_size, margin, line_gap
            )

        except Exception:
            fits = False

        if fits:
            best = (mid, lines, font, line_height, total_height)
            left = mid + 1
        else:
            right = mid - 1

    if best is None:
        # 如果连最小字号都放不下，就用最小字号并接受它
        try:
            fits, lines, font, line_height, total_height = layout_for_font_size(
                text, font_path, 10, image_size, margin, line_gap
            )
            best = (10, lines, font, line_height, total_height)
            if not fits:
                print(f"⚠️ 警告: 文本 '{text[:30]}...' 即使使用最小字号也无法容纳。")
        except Exception as e:
            raise ValueError(f"Cannot render text: {text[:80]}... Error: {e}")

    # 最终兜底检查（宽/高）
    while True:
        font_size, lines, font, line_height, total_height = best
        img = Image.new("RGB", (image_size, image_size), color=background_color)
        draw = ImageDraw.Draw(img)
        font = get_font(font_path, font_size)  # 确保使用最终的字号获取字体

        max_width = image_size - 2 * margin
        max_height = image_size - 2 * margin

        too_wide = any(measure_width(draw, font, line) > max_width for line in lines)
        too_tall = total_height > max_height

        if (too_wide or too_tall) and font_path != "default":  # 位图字体不能缩小
            font_size -= 1
            if font_size < 10:
                break  # 别缩了
            try:
                fits, lines, font, line_height, total_height = layout_for_font_size(
                    text, font_path, font_size, image_size, margin, line_gap
                )

                if not fits:
                    continue
                best = (font_size, lines, font, line_height, total_height)
                continue
            except Exception:
                continue

        # 绘制（垂直居中）
        y = margin + (max_height - total_height) // 2
        x = margin
        for i, line in enumerate(lines):
            # print(f"  line{i} width={measure_width(draw, font, line)} text='{line}'")
            draw.text((x, y), line, fill=text_color, font=font)
            y += line_height
            if i != len(lines) - 1:
                y += line_gap

        return img

    raise ValueError("Text rendering failed after fallback.")


def process_item(entry, label_map, output_root):
    try:
        image_path = entry["image_path"]
        label = str(entry["label"])
        name = label_map.get(label, None)
        if not name:
            # print(f"⚠️ Label {label} not found in label_map.")
            return 0

        image = render_text_with_adaptive_font(
            text=name,
            font_path=FONT_PATH,  # 注意：这里还是读取的全局FONT_PATH
            image_size=IMAGE_SIZE,
            background_color=(255, 255, 255),
            text_color=(0, 0, 0),
            margin=20
        )

        label_dir = os.path.join(output_root, label)
        os.makedirs(label_dir, exist_ok=True)

        base_name = os.path.basename(image_path)
        save_name = os.path.splitext(base_name)[0] + "_render.JPEG"
        save_path = os.path.join(label_dir, save_name)
        image.save(save_path, format="JPEG")
        return 1

    except Exception as e:
        # print(f"⚠️ Error rendering {entry}: {e}")
        return 0


def main():
    # ... (这部分代码在测试中被跳过) ...
    pass


# ==================================================================
# === 新增的测试辅助函数 (包含Linux特定路径) ===
# ==================================================================
def find_system_font() -> str:
    """尝试在不同操作系统上查找一个默认的 .ttf/.ttc 字体文件"""
    # 1. Windows
    if os.name == 'nt':
        font_paths = [
            "C:\\Windows\\Fonts\\Arial.ttf",
            "C:\\Windows\\Fonts\\Verdana.ttf",
            "C:\\Windows\\Fonts\\msyh.ttc"  # 微软雅黑 (中文)
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                return font_path

    # 2. macOS
    elif os.name == 'posix' and "darwin" in os.uname().sysname.lower():
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/PingFang.ttc"  # 苹方 (中文)
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                return font_path

    # 3. Linux (更通用)
    elif os.name == 'posix':
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"  # 文泉驿 (中文)
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                return font_path

    # 4. 兜底
    print("⚠️ 警告: 未找到常见的系统字体。")
    return "default"  # 将触发 get_font 中的默认加载逻辑


# ==================================================================
# === 修改后的 __main__ 入口点 (测试代码) ===
# ==================================================================

if __name__ == "__main__":
    # --- 原本的 main() 调用被注释掉 ---
    # main()

    # --- 以下是新增的测试代码 ---
    print("--- 启动 render_text_with_adaptive_font 函数测试 ---")

    # 1. 自动查找字体
    TEST_FONT_PATH = find_system_font()

    if TEST_FONT_PATH != "default":
        print(f"✅ 成功找到字体: {TEST_FONT_PATH}")
    else:
        print("将尝试使用 PIL 的默认位图字体（效果可能不佳）。")

    # 2. 定义测试用例
    test_texts = [
        "tench",
        "stoplight, traffic light, traffic signal",
        "This is a relatively long sentence that should demonstrate the wrapping feature of the layout engine.",
        "MyCamelCase/file_name.py",
        "Pneumonoultramicroscopicsilicovolcanoconiosis",
        "这是一个包含中文的长句子，它也应该能够被正确地换行处理。",
        "__underscores__ and /slashes/."
    ]

    # 3. 设置渲染参数
    TEST_IMAGE_SIZE = 512
    TEXT_COLOR = (0, 0, 0)  # 黑色
    BG_COLOR = (255, 255, 255)  # 白色
    MARGIN = 20  # 边距（与您 process_item 中的设置保持一致）

    # 4. 循环执行并保存
    for i, text in enumerate(test_texts):
        print(f"\n🎨 正在渲染 (Test {i + 1}): '{text[:50]}...'")
        try:
            image = render_text_with_adaptive_font(
                text=text,
                font_path=TEST_FONT_PATH,  # <--- 在这里传入我们找到的字体
                image_size=TEST_IMAGE_SIZE,
                text_color=TEXT_COLOR,
                background_color=BG_COLOR,
                margin=MARGIN,
                line_gap=4
            )

            save_name = f"test_render_{i + 1}.jpg"
            image.save(save_name)
            print(f"👍 成功! 图像已保存到: {os.path.abspath(save_name)}")

        except Exception as e:
            print(f"❌ 渲染失败 (Test {i + 1}): {e}")
            import traceback

            traceback.print_exc()

    print("\n--- 测试完成 ---")