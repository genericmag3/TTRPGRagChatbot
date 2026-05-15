import streamlit as st
import streamlit.components.v1 as components

_SCROLL_RESTORE_JS = """
<script>
(function() {
    var STORAGE_KEY = 'dandd_editor_scroll';

    function getTextarea() {
        return window.parent.document.querySelector('.stTextArea textarea')
               || window.parent.document.querySelector('textarea');
    }

    function onTextareaReady(ta) {
        var saved = localStorage.getItem(STORAGE_KEY);
        if (saved !== null) {
            ta.scrollTop = parseInt(saved, 10);
        }
        if (!ta.dataset.scrollListenerAttached) {
            ta.dataset.scrollListenerAttached = 'true';
            ta.addEventListener('scroll', function() {
                localStorage.setItem(STORAGE_KEY, this.scrollTop);
            }, { passive: true });
        }
    }

    // Poll until the textarea is in the parent DOM, then restore immediately.
    var attempts = 0;
    var interval = setInterval(function() {
        var ta = getTextarea();
        if (ta || attempts >= 20) {
            clearInterval(interval);
            if (ta) onTextareaReady(ta);
        }
        attempts++;
    }, 50);
})();
</script>
"""


class TextEditorHandler:
    def render(
        self,
        key: str,
        initial_value: str = "",
        height: int = 600,
        font_family: str = "Georgia",
        font_size: int = 16,
    ) -> str:
        st.markdown(
            f"""
            <style>
            .stTextArea textarea {{
                font-family: '{font_family}', serif !important;
                font-size: {font_size}px !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        components.html(_SCROLL_RESTORE_JS, height=0)
        return st.text_area(
            label="Notes",
            value=initial_value,
            height=height,
            key=key,
            label_visibility="collapsed",
        )
