"""Thin wrapper so the HTTP layer never imports the renderer at module scope."""

from __future__ import annotations


def summarise(path):
    from gnv.convert.volume_meshes import volume_summary

    return volume_summary(path)
