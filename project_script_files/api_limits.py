"""
API Usage Limter Module
Handles API usage tracking and limiting for both individual users and total usage.
Uses file-based storage for persistence and Streamlit session state for runtime tracking.
"""
import json
import os
from datetime import datetime
import streamlit as st
from configs import config

class APILimiter:
    """Handles API usage limits and tracking."""

    def __init__(self, 
                daily_user_limit: int = config.DAILY_USER_LIMIT,
                daily_total_limit: int = config.DAILY_TOTAL_LIMIT,
                storage_path: str = config.API_USAGE_PATH):
        """
        Initialize API limiter.
        
        Args:
            daily_user_limit: Maximum calls per user per day
            daily_total_limit: Maximum total calls per day
            storage_path: Path to usage storage file
        """
        self.daily_user_limit = daily_user_limit
        self.daily_total_limit = daily_total_limit
        self.storage_path = storage_path
        self.usage_data = self._load_usage_data()
        
        # Initialize session state for user tracking
        if 'user_api_calls' not in st.session_state:
            st.session_state.user_api_calls = 0
            
    def _load_usage_data(self) -> dict:
        """Load usage data from storage file."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            return {'date': datetime.now().strftime('%Y-%m-%d'), 'total_calls': 0}
        except Exception as e:
            st.error(f"Error loading API usage data: {str(e)}")
            return {'date': datetime.now().strftime('%Y-%m-%d'), 'total_calls': 0}

    def _save_usage_data(self):
        """Save usage data to storage file."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.usage_data, f)
        except Exception as e:
            st.error(f"Error saving API usage data: {str(e)}")

    def _reset_if_new_day(self):
        """Reset counters if it's a new day."""
        current_date = datetime.now().strftime('%Y-%m-%d')
        if self.usage_data['date'] != current_date:
            self.usage_data = {'date': current_date, 'total_calls': 0}
            st.session_state.user_api_calls = 0
            self._save_usage_data()

    def check_limits(self) -> bool:
        """
        Check if API call is allowed under current limits.
        
        Returns:
            bool: True if call is allowed, False otherwise
        """
        self._reset_if_new_day()
        
        # Check user limit
        if st.session_state.user_api_calls >= self.daily_user_limit:
            st.warning(f"You've reached your daily limit of {self.daily_user_limit} API calls. "
                        "Please try again tomorrow.")
            return False
        
        # Check total limit
        if self.usage_data['total_calls'] >= self.daily_total_limit:
            st.error(f"The total daily limit of {self.daily_total_limit} API calls has been reached. "
                    "Please try again tomorrow.")
            return False
        
        return True

    def increment_usage(self):
        """Increment API usage counters."""
        st.session_state.user_api_calls += 1
        self.usage_data['total_calls'] += 1
        self._save_usage_data()

    def display_usage_stats(self):
        """Display current API usage statistics."""
        self._reset_if_new_day()
        
        st.sidebar.write("### API Usage Statistics")
        st.sidebar.write(f"Your usage today: {st.session_state.user_api_calls}/{self.daily_user_limit}")
        st.sidebar.write(f"Total usage today: {self.usage_data['total_calls']}/{self.daily_total_limit}")
        st.sidebar.progress(st.session_state.user_api_calls / self.daily_user_limit)