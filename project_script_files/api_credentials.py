"""
API Credentials Manager Module
Handles user-provided API credentials and determines whether to use default or user credentials.
"""
import streamlit as st
import os
from typing import Dict, Optional

class APICredentialsManager:
    """Manages API credentials for different services."""

    def __init__(self):
        """Initialize the credentials manager."""
        # Initialize session state for credentials
        if 'using_own_credentials' not in st.session_state:
            st.session_state.using_own_credentials = {
                'aws': False,
                'openai': False
            }
        if 'user_credentials' not in st.session_state:
            st.session_state.user_credentials = {
                'aws': {},
                'openai': {}
            }

    def clear_credentials(self):
        """Clear all user credentials from session state."""
        st.session_state.using_own_credentials = {
            'aws': False,
            'openai': False
        }
        st.session_state.user_credentials = {
            'aws': {},
            'openai': {}
        }

    def credentials_ui(self) -> None:
        """Displays the credentials management UI in the sidebar."""
        with st.sidebar:
            st.write("### API Credentials")

            # Add clear credentials button
            if st.button("Clear Saved Credentials"):
                self.clear_credentials()
                st.rerun()

            # OpenAI Credentials
            st.write("#### OpenAI Settings")
            use_own_openai = st.checkbox(
                "Use my own OpenAI API key",
                value=st.session_state.using_own_credentials['openai']
            )
            
            if use_own_openai:
                openai_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    value=st.session_state.user_credentials['openai'].get('api_key', '')
                )
                if openai_key:
                    st.session_state.user_credentials['openai']['api_key'] = openai_key
                    st.session_state.using_own_credentials['openai'] = True
                else:
                    st.session_state.using_own_credentials['openai'] = False
            
            # AWS Credentials
            st.write("#### AWS Settings")
            use_own_aws = st.checkbox(
                "Use my own AWS credentials",
                value=st.session_state.using_own_credentials['aws']
            )
            
            if use_own_aws:
                aws_access_key = st.text_input(
                    "AWS Access Key ID",
                    type="password",
                    value=st.session_state.user_credentials['aws'].get('access_key_id', '')
                )
                aws_secret_key = st.text_input(
                    "AWS Secret Access Key",
                    type="password",
                    value=st.session_state.user_credentials['aws'].get('secret_key', '')
                )
                aws_region = st.text_input(
                    "AWS Region",
                    value=st.session_state.user_credentials['aws'].get('region', 'eu-west-1')
                )
                
                if aws_access_key and aws_secret_key:
                    st.session_state.user_credentials['aws'].update({
                        'access_key_id': aws_access_key,
                        'secret_key': aws_secret_key,
                        'region': aws_region
                    })
                    st.session_state.using_own_credentials['aws'] = True
                else:
                    st.session_state.using_own_credentials['aws'] = False

    def get_credentials(self, service: str) -> Dict[str, str]:
        """Get the appropriate credentials for a service.
        
        Args:
            service: Service name ('aws' or 'openai')
            
        Returns:
            dict: Credentials for the specified service
        """
        if st.session_state.using_own_credentials.get(service):
            if service == 'aws':
                creds = {
                    'aws_access_key_id': st.session_state.user_credentials['aws']['access_key_id'],
                    'aws_secret_access_key': st.session_state.user_credentials['aws']['secret_key'],
                    'region_name': st.session_state.user_credentials['aws']['region']
                }
                # Clear credentials after use
                self.clear_service_credentials('aws')
                return creds
            elif service == 'openai':
                creds = {
                    'api_key': st.session_state.user_credentials['openai']['api_key']
                }
                # Clear credentials after use
                self.clear_service_credentials('openai')
                return creds
        
        # Return default credentials
        if service == 'aws':
            return {
                'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
                'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
                'region_name': os.getenv('AWS_REGION', 'eu-west-1')
            }
        elif service == 'openai':
            return {
                'api_key': os.getenv('OPENAI_API_KEY')
            }
        
        return {}
    
    def clear_service_credentials(self, service: str):
        """Clear credentials for a specific service.
        
        Args:
            service: Service name ('aws' or 'openai')
        """
        st.session_state.using_own_credentials[service] = False
        st.session_state.user_credentials[service] = {}

    def is_using_own_credentials(self, service: str) -> bool:
        """Check if using user-provided credentials for a service.
        
        Args:
            service: Service name ('aws' or 'openai')
            
        Returns:
            bool: True if using user credentials, False otherwise
        """
        return st.session_state.using_own_credentials.get(service, False)