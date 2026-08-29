#!/bin/sh

ollama serve &

sleep 5

ollama pull your-model-name

wait