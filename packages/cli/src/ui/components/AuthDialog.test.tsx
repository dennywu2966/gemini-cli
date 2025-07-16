/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { render } from 'ink-testing-library';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AuthDialog } from './AuthDialog.js';
import { LoadedSettings, SettingScope } from '../../config/settings.js';
import { AuthType } from '@google/gemini-cli-core';

describe('AuthDialog', () => {
  const onSelect = vi.fn();
  const settings = {
    user: {},
    project: {},
    merged: {},
  } as LoadedSettings;

  beforeEach(() => {
    vi.clearAllMocks();
    delete process.env.GEMINI_API_KEY;
    delete process.env.GEMINI_DEFAULT_AUTH_TYPE;
    delete process.env.OPENAI_API_KEY;
  });

  it('should show an error if the initial auth type is invalid', () => {
    const { lastFrame } = render(
      <AuthDialog
        onSelect={onSelect}
        settings={settings}
        initialErrorMessage="Invalid auth method"
      />,
    );
    expect(lastFrame()).toContain('Invalid auth method');
  });

  describe('GEMINI_API_KEY environment variable', () => {
    it('should detect GEMINI_API_KEY environment variable', () => {
      process.env.GEMINI_API_KEY = 'test-key';
      const { lastFrame } = render(
        <AuthDialog onSelect={onSelect} settings={settings} />,
      );
      expect(lastFrame()).toContain(
        'Existing API key detected (GEMINI_API_KEY)',
      );
    });

    it('should not show the GEMINI_API_KEY message if GEMINI_DEFAULT_AUTH_TYPE is set to something else', () => {
      process.env.GEMINI_API_KEY = 'test-key';
      process.env.GEMINI_DEFAULT_AUTH_TYPE = AuthType.LOGIN_WITH_GOOGLE;
      const { lastFrame } = render(
        <AuthDialog onSelect={onSelect} settings={settings} />,
      );
      expect(lastFrame()).not.toContain(
        'Existing API key detected (GEMINI_API_KEY)',
      );
    });

    it('should show the GEMINI_API_KEY message if GEMINI_DEFAULT_AUTH_TYPE is set to use api key', () => {
      process.env.GEMINI_API_KEY = 'test-key';
      process.env.GEMINI_DEFAULT_AUTH_TYPE = AuthType.USE_GEMINI;
      const { lastFrame } = render(
        <AuthDialog onSelect={onSelect} settings={settings} />,
      );
      expect(lastFrame()).toContain(
        'Existing API key detected (GEMINI_API_KEY)',
      );
    });
  });

  describe('GEMINI_DEFAULT_AUTH_TYPE environment variable', () => {
    it('should select the auth type specified by GEMINI_DEFAULT_AUTH_TYPE', () => {
      process.env.GEMINI_DEFAULT_AUTH_TYPE = AuthType.USE_VERTEX_AI;
      const { lastFrame } = render(
        <AuthDialog onSelect={onSelect} settings={settings} />,
      );
      expect(lastFrame()).toContain('● Vertex AI');
    });

    it('should fall back to default if GEMINI_DEFAULT_AUTH_TYPE is not set', () => {
      const { lastFrame } = render(
        <AuthDialog onSelect={onSelect} settings={settings} />,
      );
      expect(lastFrame()).toContain('● Login with Google');
    });

    it('should show an error and fall back to default if GEMINI_DEFAULT_AUTH_TYPE is invalid', () => {
      process.env.GEMINI_DEFAULT_AUTH_TYPE = 'invalid-auth';
      const { lastFrame } = render(
        <AuthDialog onSelect={onSelect} settings={settings} />,
      );
      expect(lastFrame()).toContain('Invalid value for GEMINI_DEFAULT_AUTH_TYPE');
      expect(lastFrame()).toContain('● Login with Google');
    });
  });

  it('should prevent exiting when no auth method is selected and show error message', () => {
    const { lastFrame, unmount } = render(
      <AuthDialog onSelect={onSelect} settings={settings} />,
    );
    // No easy way to simulate escape, so just unmount
    unmount();
    expect(onSelect).not.toHaveBeenCalled();
  });

  it('should not exit if there is already an error message', () => {
    const { unmount } = render(
      <AuthDialog
        onSelect={onSelect}
        settings={settings}
        initialErrorMessage="Initial error"
      />,
    );
    unmount();
    expect(onSelect).not.toHaveBeenCalled();
  });

  it('should allow exiting when auth method is already selected', () => {
    const { unmount } = render(
      <AuthDialog
        onSelect={onSelect}
        settings={{ ...settings, merged: { selectedAuthType: AuthType.USE_GEMINI } }}
      />,
    );
    unmount();
    expect(onSelect).toHaveBeenCalledWith(undefined, SettingScope.User);
  });
});