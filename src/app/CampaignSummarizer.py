import re
import json
import streamlit as st
from streamlit_lottie import st_lottie


class CampaignSummarizer:
    def __init__(self, llm_handler, summary_handler):
        self.llm_handler = llm_handler
        self.summary_handler = summary_handler

    def run(self):
        """Entry point for the campaign summary Streamlit page."""
        if not st.session_state.get("model_name"):
            st.title("📖 Campaign Summary")
            st.info("Please select a model in **Model Options** on the Q&A page before viewing the campaign summary.")
            st.stop()

        if not self.summary_handler.raw_notes_exist():
            st.title("📖 Campaign Summary")
            st.info("No campaign notes found. Please upload your notes on the Q&A page first.")
            st.stop()

        party_members = st.session_state.get("party_members", []) or []
        named_members = [m for m in party_members if m.get("name", "").strip()]
        if not named_members:
            st.title("📖 Campaign Summary")
            st.info("Please define your party members on the Q&A page before generating a campaign summary.")
            st.stop()

        existing = self.summary_handler.get_saved_summary()

        if existing and not st.session_state.get("_regenerating_summary"):
            self.__render_existing_summary(existing)
            st.stop()

        st.title("📖 Campaign Summary")

        if not existing:
            st.info("No campaign summary has been generated yet.")
            st.warning(
                "Generating a summary may take several minutes depending on the length "
                "of your notes and your hardware. Please be patient."
            )
            if not st.button("✨ Generate Campaign Summary", type="primary"):
                st.stop()

        self.__generate_and_display()

    def __render_existing_summary(self, existing):
        generated_date = existing.get("generated_at", "")[:10]
        model_used = existing.get("model", "unknown model")

        header_col, btn_col = st.columns([5, 1])
        with header_col:
            st.title("📖 Campaign Summary")
        with btn_col:
            st.write("")
            if st.button("🔄 Regenerate", use_container_width=True):
                st.session_state._regenerating_summary = True
                st.rerun()

        self.__render_summary(existing["summary"], model_used, generated_date)

    def __extract_headers(self, text):
        return [
            (len(m.group(1)), m.group(2).strip())
            for m in re.finditer(r'^(#{1,3})\s+(.+)$', text, re.MULTILINE)
        ]

    def __render_summary(self, summary_text, model_used, generated_date):
        headers = self.__extract_headers(summary_text)

        with st.sidebar:
            if headers:
                st.markdown("**Contents**")
                for level, title in headers:
                    indent = "&nbsp;" * (4 * (level - 1))
                    st.markdown(f"{indent}• {title}", unsafe_allow_html=True)
                st.divider()
            st.caption(f"Model: **{model_used}**")
            st.caption(f"Generated: {generated_date}")

        st.markdown(summary_text)

    def __generate_and_display(self):
        model_name = st.session_state.model_name
        model_temp = st.session_state.get("model_temperature", 0.7)
        party_members = st.session_state.get("party_members", [])

        try:
            self.llm_handler.load_model(str(model_name), float(model_temp))
        except Exception as e:
            st.error(f"Could not load model **{model_name}**: {e}")
            st.session_state.pop("_regenerating_summary", None)
            st.stop()

        try:
            with open("assets/star-magic.json", "r", errors="ignore") as f:
                magic_spinner = json.load(f)
        except Exception:
            magic_spinner = None

        animation_slot = st.empty()
        progress_slot = st.empty()

        if magic_spinner:
            with animation_slot.container():
                st_lottie(magic_spinner, height=200, key="summary_page_spinner")

        progress_bar = progress_slot.progress(0, text="Starting summary generation...")

        final_summary = None
        try:
            for is_done, progress, text in self.summary_handler.generate_summary_streaming(model_name, party_members):
                if is_done:
                    final_summary = text
                    st.session_state.summary_generated = True
                else:
                    progress_bar.progress(progress / 100, text=text)
        except Exception as e:
            animation_slot.empty()
            progress_slot.empty()
            st.session_state.pop("_regenerating_summary", None)
            st.error(f"Summary generation failed: {e}")
            st.stop()

        animation_slot.empty()
        progress_slot.empty()
        st.session_state.pop("_regenerating_summary", None)

        if final_summary:
            st.success("Campaign summary generated!")
            saved = self.summary_handler.get_saved_summary()
            self.__render_summary(
                final_summary,
                saved.get("model", model_name),
                saved.get("generated_at", "")[:10],
            )
        else:
            st.error("Summary generation returned no content.")
