// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { resolve } from 'path';

function parseVersion(v: string): [number, number, number] {
  const parts = v.replace(/^[^0-9]*/, '').split('.').map(Number);
  return [parts[0] ?? 0, parts[1] ?? 0, parts[2] ?? 0];
}

function gte(a: string, b: string): boolean {
  const [a0, a1, a2] = parseVersion(a);
  const [b0, b1, b2] = parseVersion(b);
  if (a0 !== b0) return a0 > b0;
  if (a1 !== b1) return a1 > b1;
  return a2 >= b2;
}

function readInstalledVersion(pkg: string): string | null {
  try {
    const pkgJsonPath = resolve(__dirname, '../../../..', 'node_modules', pkg, 'package.json');
    const pkgJson = JSON.parse(readFileSync(pkgJsonPath, 'utf-8')) as { version?: string };
    return pkgJson.version ?? null;
  } catch {
    return null;
  }
}

describe('security version pins', () => {
  it('vite >= 6.4.3 (CVE-2026-53632 / GHSA-v6wh-96g9-6wx3)', () => {
    const v = readInstalledVersion('vite');
    expect(v, 'vite must be installed in node_modules').not.toBeNull();
    expect(gte(v!, '6.4.3'), `vite@${v} must be >= 6.4.3`).toBe(true);
  });

  it('vitest >= 3.2.6 (GHSA-5xrq-8626-4rwp)', () => {
    const v = readInstalledVersion('vitest');
    expect(v, 'vitest must be installed in node_modules').not.toBeNull();
    expect(gte(v!, '3.2.6'), `vitest@${v} must be >= 3.2.6`).toBe(true);
  });
});
