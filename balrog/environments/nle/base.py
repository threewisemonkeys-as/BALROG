import random

from nle import nle_language_obsv
from nle.language_wrapper.wrappers import nle_language_wrapper as language_wrapper
from nle.nethack import USEFUL_ACTIONS
from PIL import Image

from balrog.environments import Strings

from ..minihack import ACTIONS as MINIHACK_ACTIONS
from .progress import get_progress_system
from .render import tty_render_image
from .render_rgb import rgb_render_image
from .perc import compute_percepts, pretty


def decode_grid(screen_desc):
    if hasattr(screen_desc, "cpu"):
        screen_desc = screen_desc.detach().cpu().numpy()
    H, W, L = screen_desc.shape  # L should be 80
    out = [["" for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            out[r][c] = bytes(screen_desc[r, c]).split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    return out

def trim_empty_rows_cols(text_grid, empty_tokens=("", " ", "unknown", "void"), min_keep_rows=1, min_keep_cols=1):
    H = len(text_grid); W = len(text_grid[0]) if H else 0
    def is_empty_cell(s):
        if s is None: return True
        s2 = s.strip().lower()
        return (s2 == "") or (s2 in empty_tokens)
    def is_empty_row(r): return all(is_empty_cell(text_grid[r][c]) for c in range(W))
    def is_empty_col(c): return all(is_empty_cell(text_grid[r][c]) for r in range(H))
    top = 0
    while top < H and is_empty_row(top): top += 1
    bottom = H - 1
    while bottom >= 0 and is_empty_row(bottom): bottom -= 1
    left = 0
    while left < W and is_empty_col(left): left += 1
    right = W - 1
    while right >= 0 and is_empty_col(right): right -= 1
    if top > bottom:
        mid = H // 2; half = max(0, (min_keep_rows - 1) // 2)
        top = max(0, mid - half); bottom = min(H - 1, top + min_keep_rows - 1)
    if left > right:
        mid = W // 2; half = max(0, (min_keep_cols - 1) // 2)
        left = max(0, mid - half); right = min(W - 1, left + min_keep_cols - 1)
    return top, bottom, left, right

def render_grid_trimmed(text_grid, pad=18, max_cell_chars=18, sep="|", empty_tokens=("", " ", "unknown", "void")):
    top, bottom, left, right = trim_empty_rows_cols(text_grid, empty_tokens=empty_tokens)
    top += 1
    bottom += 1
    def fmt_cell(s):
        s = "" if s is None else s
        s = " ".join(s.split())
        if len(s) > max_cell_chars: s = s[: max_cell_chars - 1] + "…"
        return s.ljust(pad)
    render_str = f"Showing rows [{top}..{bottom}] cols [{left}..{right}] (from {len(text_grid)}x{len(text_grid[0]) if text_grid else 0})\n\n"
    for r in range(top, bottom + 1):
        row = (f" {sep} ").join(fmt_cell(text_grid[r][c]) for c in range(left, right + 1))
        render_str += f"{r:3d} | {row}\n"
    return render_str


class NLELanguageWrapper(language_wrapper.NLELanguageWrapper):
    def __init__(self, env, vlm=False, include_lang_obs=True, include_perc_obs=False, use_textual_desc=False, hide_obs_when_image=False):
        super().__init__(env, use_language_action=True)
        self.nle_language = nle_language_obsv.NLELanguageObsv()
        self.language_action_space = self.create_action_space()
        self.env = env
        self.vlm = vlm
        self.hide_obs_when_image = hide_obs_when_image
        self.done = False

        if vlm and hide_obs_when_image:
            self.prompt_mode = "language"
        else:
            self.prompt_mode = "hybrid"

        self.progress = get_progress_system(self.env)
        self.max_steps = self.env.unwrapped._max_episode_steps
        self.include_lang_obs = include_lang_obs
        self.include_perc_obs = include_perc_obs
        self.use_textual_desc = use_textual_desc

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.done = done if not self.done else self.done
        self.progress.update(obs["obs"], reward, self.done, info)
        step_info = dict(info)
        step_info.update(self.progress.__dict__)
        return obs, reward, self.done, step_info

    def post_reset(self, obsv):
        return self.post_step(obsv)

    def reset(self, **kwargs):
        self.progress = get_progress_system(self.env)
        obsv = self.env.reset(**kwargs)
        return self.post_reset(obsv)

    def post_step(self, nle_obsv):
        return self.nle_process_obsv(nle_obsv)

    @property
    def default_action(self):
        if "minihack" in self.env.spec.id.lower():
            return "north"
        else:
            return "esc"

    def get_text_action(self, action):
        return NLELanguageWrapper.all_nle_action_map[self.env.actions[action]][0]

    def nle_process_obsv(self, nle_obsv):
        img = Image.fromarray(self.render("tiles")).convert("RGB") if self.vlm else None
        text = self.nle_obsv_type(nle_obsv)

        return {
            "text": text,
            "image": img,
            "obs": nle_obsv,
        }

    def nle_obsv_type(self, nle_obsv):
        nle_obsv = self.nle_obsv_to_language(nle_obsv)
        if self.prompt_mode == "language":
            return self.render_text(nle_obsv)
        elif self.prompt_mode == "hybrid":
            return self.render_hybrid(nle_obsv)
        else:
            raise ValueError(f'"{self.prompt_mode}" is not a valid prompt mode.')

    def render(self, mode="human"):
        if mode == "tiles":
            obs = self.env.unwrapped.last_observation
            glyphs = obs[self.env.unwrapped._observation_keys.index("glyphs")]
            return rgb_render_image(glyphs)
        elif mode == "tty_image":
            obs = self.env.unwrapped.last_observation
            tty_chars = obs[self.env.unwrapped._observation_keys.index("tty_chars")]
            tty_colors = obs[self.env.unwrapped._observation_keys.index("tty_colors")]
            return tty_render_image(tty_chars, tty_colors)
        else:
            return super().render(mode)

    def get_stats(self):
        return self.progress.__dict__

    def create_action_space(self):
        if "minihack" in self.env.spec.id.lower():
            available_actions = {}
            for action in self.env.actions:
                action_key = NLELanguageWrapper.all_nle_action_map[action][0]
                if action_key not in MINIHACK_ACTIONS:
                    continue
                available_actions[action_key] = MINIHACK_ACTIONS[action_key]

            all_actions = [action for action, _ in available_actions.items()]

        else:
            available_actions = [
                action_strs[0]
                for action, action_strs in NLELanguageWrapper.all_nle_action_map.items()
                if action in USEFUL_ACTIONS
            ]
            single_chars = [chr(i) for i in range(ord("a"), ord("z") + 1)] + [
                chr(i) for i in range(ord("A"), ord("Z") + 1)
            ]
            single_digits = [str(i) for i in range(10)]
            double_digits = [f"{i:02d}" for i in range(100)]
            all_actions = available_actions + single_chars + single_digits + double_digits

        return Strings(all_actions)

    def ascii_render(self, chars):
        rows, cols = chars.shape
        result = ""
        for i in range(rows):
            for j in range(cols):
                entry = chr(chars[i, j])
                result += entry
            result += "\n"
        return result

    def nle_obsv_to_language(self, nle_obsv):
        """Translate NLE Observation into a language observation.
        Args:
            nle_obsv (dict): NLE observation from the base environment
        Returns:
            (dict): language observation
        """

        message = (
            nle_obsv["text_message"]
            if "text_message" in nle_obsv
            else self.nle_language.text_message(nle_obsv["tty_chars"]).decode("latin-1")
        )

        glyphs = nle_obsv["glyphs"]
        blstats = nle_obsv["blstats"]
        tty_cursor = nle_obsv["tty_cursor"]
        inv_strs = nle_obsv["inv_strs"]
        inv_letters = nle_obsv["inv_letters"]

        return {
            "text_glyphs": self.nle_language.text_glyphs(glyphs, blstats).decode("latin-1"),
            "text_message": message,
            "text_blstats": self.nle_language.text_blstats(blstats).decode("latin-1"),
            "text_inventory": self.nle_language.text_inventory(inv_strs, inv_letters).decode("latin-1"),
            "text_cursor": self.nle_language.text_cursor(glyphs, blstats, tty_cursor).decode("latin-1"),
            "tty_chars": nle_obsv["tty_chars"],
            "tty_cursor": nle_obsv["tty_cursor"],
            **({"screen_descriptions": nle_obsv["screen_descriptions"]} if "screen_descriptions" in nle_obsv else {}),
        }

    def render_text(self, nle_obsv):
        long_term_observations = [("text_message", "message")]
        if self.include_lang_obs:
            long_term_observations.append(("text_glyphs", "language observation"))
        long_term_observations.extend([
            ("text_cursor", "cursor"),
        ])

        short_term_observations = [
            ("text_blstats", "statistics"),
            ("text_inventory", "inventory"),
        ]

        # Add perceptual observations if enabled
        if self.include_perc_obs:
            ascii_map = self.ascii_render(nle_obsv["tty_chars"])
            try:
                percepts = compute_percepts(ascii_map, doors_block=True)
                perc_text = pretty(percepts)
                nle_obsv["perceptual_features"] = perc_text
                short_term_observations.append(("perceptual_features", "perceptual features"))
            except Exception as e:
                pass

        long_term_context = "\n".join([f"{name}:\n{nle_obsv[key]}\n" for key, name in long_term_observations])
        short_term_context = "\n".join([f"{name}:\n{nle_obsv[key]}\n" for key, name in short_term_observations])

        return {
            "long_term_context": long_term_context,
            "short_term_context": short_term_context,
        }

    def render_hybrid(self, nle_obsv):
        ascii_map = self.ascii_render(nle_obsv["tty_chars"])
        map_display = "\n".join(ascii_map.split("\n")[1:])  # remove first line
        
        if self.use_textual_desc and "screen_descriptions" in nle_obsv:
            grid_text = decode_grid(nle_obsv["screen_descriptions"])
            map_display += '\n\nmap with descriptions:\n' + render_grid_trimmed(grid_text, pad=1)
        
        cursor = nle_obsv["tty_cursor"]
        cursor = f"(x={cursor[1]}, y={cursor[0]})"

        nle_obsv["map"] = map_display
        nle_obsv["text_cursor"] = nle_obsv["text_cursor"] + "\n" + cursor

        long_term_observations = [("text_message", "message")]
        if self.include_lang_obs:
            long_term_observations.append(("text_glyphs", "language observation"))
        long_term_observations.extend([
            ("text_cursor", "cursor"),
            ("map", "map"),
        ])
        short_term_observation = [
            ("text_inventory", "inventory"),
        ]

        # Add perceptual observations if enabled
        if self.include_perc_obs:
            try:
                ascii_map_for_percept = '\n'.join(ascii_map.strip().splitlines()[1:-2])
                percepts = compute_percepts(ascii_map_for_percept, doors_block=True)
                perc_text = pretty(percepts)
                nle_obsv["perceptual_features"] = perc_text
                short_term_observation.append(("perceptual_features", "perceptual features"))
            except Exception as e:
                pass

        long_term_context = "\n".join([f"{name}:\n{nle_obsv[key]}\n" for key, name in long_term_observations])
        short_term_context = "\n".join([f"{name}:\n{nle_obsv[key]}\n" for key, name in short_term_observation])

        return {
            "long_term_context": long_term_context,
            "short_term_context": short_term_context,
        }
