// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// This file is derived from Articraft (https://github.com/mattzh72/articraft),
// copyright the Articraft contributors, licensed under the Apache License, Version 2.0,
// and has been modified by NVIDIA.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import json from "react-syntax-highlighter/dist/esm/languages/hljs/json";
import python from "react-syntax-highlighter/dist/esm/languages/hljs/python";
import xml from "react-syntax-highlighter/dist/esm/languages/hljs/xml";
import { atomOneLight } from "react-syntax-highlighter/dist/esm/styles/hljs";

export type HighlightLanguage = "json" | "python" | "xml";

SyntaxHighlighter.registerLanguage("json", json);
SyntaxHighlighter.registerLanguage("python", python);
SyntaxHighlighter.registerLanguage("xml", xml);

export const codeTheme = {
  ...atomOneLight,
  hljs: {
    ...atomOneLight.hljs,
    background: "transparent",
    color: "#1f2937",
  },
};

export { SyntaxHighlighter };
