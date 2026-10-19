"""Verification runs must never contact Telegram when delivery is disabled."""
from unittest.mock import patch
import alert_utils


def test_disabled_telegram_does_not_contact_network(monkeypatch):
    monkeypatch.setenv('TELEGRAM_DISABLED','true')
    with patch.object(alert_utils.requests,'post') as post:
        assert alert_utils.send_telegram_safe('https://api.telegram.org/botexample/sendMessage',
                                              data={'chat_id':'example','text':'test'}) is False
        post.assert_not_called()


def test_missing_recipient_does_not_contact_network(monkeypatch):
    monkeypatch.delenv('TELEGRAM_DISABLED',raising=False)
    with patch.object(alert_utils.requests,'post') as post:
        assert alert_utils.send_telegram_safe('https://api.telegram.org/bot/sendMessage',
                                              data={'chat_id':'','text':'test'}) is False
        post.assert_not_called()
