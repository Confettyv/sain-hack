#!/bin/sh
# Песочница на mvm: выкладывается только закоммиченный HEAD, .env на сервере не трогается.
set -eu
cd "$(dirname "$0")/.."
REV=$(git rev-parse --short HEAD)
git archive HEAD | ssh mvm "set -e; rm -rf /opt/sain/app.new; mkdir -p /opt/sain/app.new; tar -x -C /opt/sain/app.new; \
  echo $REV > /opt/sain/app.new/REVISION; rm -rf /opt/sain/app.old; [ -d /opt/sain/app ] && mv /opt/sain/app /opt/sain/app.old; \
  mv /opt/sain/app.new /opt/sain/app; systemctl restart sain; sleep 2; systemctl is-active sain; \
  curl -s -m 5 http://127.0.0.1:8101/health; echo; ps -o rss= -C python3 | head -0; \
  systemctl show sain -p MemoryCurrent"
echo "выложен $REV"
