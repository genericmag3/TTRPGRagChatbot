import streamlit as st


class TextEditorHandler:
    def render(self, key: str, initial_value: str = "", height: int = 600) -> str:
        return st.text_area(
            label="Notes",
            value=initial_value,
            height=height,
            key=key,
            label_visibility="collapsed",
        )
