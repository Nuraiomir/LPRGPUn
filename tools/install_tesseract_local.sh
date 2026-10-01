#!/usr/bin/env bash
# Ставит tesseract в домашнюю папку, без sudo и без conda.
#
# Способ обычный для машин без прав: пакеты скачиваются из тех же apt-источников,
# которыми машина и так пользуется, и распаковываются в ~/.tess вместо системы.
# dpkg -x только распаковывает архив, права ему не нужны.
#
# Имена пакетов не зашиты: они разные в разных версиях Ubuntu (liblept5,
# libleptonica6 и так далее). apt-cache сам находит зависимости.
#
#   bash tools/install_tesseract_local.sh
#
# В конце скрипт напечатает три строки export. Их надо выполнить в той же
# оболочке, из которой потом запускается сравнение.
set -u

PREFIX="${HOME}/.tess"
ROOT="${PREFIX}/root"
DEBS="${PREFIX}/debs"

echo "ставлю tesseract в ${PREFIX}"
mkdir -p "${DEBS}" "${ROOT}"

echo
echo "1/4  ищу зависимости в apt-источниках"
PKGS=$(apt-cache depends --recurse --no-recommends --no-suggests \
       --no-conflicts --no-breaks --no-replaces --no-enhances \
       tesseract-ocr tesseract-ocr-eng 2>/dev/null | grep "^\w" | sort -u)
COUNT=$(echo "${PKGS}" | grep -c . || true)
if [ "${COUNT}" -lt 5 ]; then
    echo "  apt-источники недоступны: найдено пакетов ${COUNT}"
    echo "  дальше идти некуда, сеть до репозиториев закрыта"
    exit 1
fi
echo "  пакетов в списке: ${COUNT}"

echo
echo "2/4  качаю (часть уже стоит в системе, такие пропустятся)"
cd "${DEBS}" || exit 1
# shellcheck disable=SC2086
apt-get download ${PKGS} 2>/dev/null
DEB_COUNT=$(ls -1 ./*.deb 2>/dev/null | wc -l)
echo "  скачано файлов: ${DEB_COUNT}"
if [ "${DEB_COUNT}" -eq 0 ]; then
    echo "  ничего не скачалось, дальше смысла нет"
    exit 1
fi

echo
echo "3/4  распаковываю в ${ROOT}"
for d in ./*.deb; do
    dpkg -x "${d}" "${ROOT}" 2>/dev/null
done

BIN="${ROOT}/usr/bin/tesseract"
if [ ! -x "${BIN}" ]; then
    echo "  программы нет по пути ${BIN}"
    echo "  посмотри, что распаковалось: ls ${ROOT}/usr/bin | head"
    exit 1
fi

TESSDATA=$(dirname "$(find "${ROOT}/usr/share" -name eng.traineddata 2>/dev/null | head -1)")
LIBDIR="${ROOT}/usr/lib/x86_64-linux-gnu"

echo
echo "4/4  проверяю"
PATH="${ROOT}/usr/bin:${PATH}" \
LD_LIBRARY_PATH="${LIBDIR}:${LD_LIBRARY_PATH:-}" \
TESSDATA_PREFIX="${TESSDATA}" \
    "${BIN}" --version 2>&1 | head -2

echo
echo "==================================================================="
echo "Готово. Выполни эти три строки в той оболочке, где будешь запускать"
echo "сравнение (они действуют до закрытия окна):"
echo
echo "export PATH=\"${ROOT}/usr/bin:\$PATH\""
echo "export LD_LIBRARY_PATH=\"${LIBDIR}:\$LD_LIBRARY_PATH\""
echo "export TESSDATA_PREFIX=\"${TESSDATA}\""
echo
echo "Проверить, что подхватилось:  which tesseract && tesseract --version"
echo "Освободить место потом:       rm -rf ${DEBS}"
echo "==================================================================="
