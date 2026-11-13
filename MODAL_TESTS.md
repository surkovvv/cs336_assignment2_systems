# GPU-тесты CS336 через Modal

Из корня `assignment2-systems`:

```bash
uvx modal run modal_tests.py
```

Только вариант без causal mask:

```bash
uvx modal run modal_tests.py --k 'test_flash_forward_pass_triton and False'
```

Другой набор тестов:

```bash
uvx modal run modal_tests.py --test tests/test_attention.py --k test_flash_backward_triton
```

Посмотреть выбранные тесты, не выполняя их (GPU всё равно выделяется):

```bash
uvx modal run modal_tests.py --collect-only
```

Цикл работы: сохранить изменения на Mac → снова запустить команду.
Каждый запуск отправляет текущие `cs336_systems`, `tests` и `cs336_basics`.
Зависимости кешируются в Linux-образе отдельно от исходников.
Менять исходники во время теста бесполезно: текущий запуск использует снимок.
Локальное окружение `.venv`, данные, checkpoints и история Git не отправляются.
Скрипт возвращает ненулевой код завершения, если pytest падает.
Путь проекта определяется относительно `modal_tests.py`; при необходимости
его можно переопределить переменной окружения `CS336_PROJECT_DIR`.
Если Modal ещё не авторизован, сначала выполни `uvx modal setup`.

Интерактивный терминал с теми же исходниками и GPU:

```bash
uvx modal shell modal_tests.py::run_tests
```

Внутри него:

```bash
cd /workspace/assignment2-systems
.venv/bin/python -m pytest tests/test_attention.py -k test_flash_forward_pass_triton -vv
```

В этом варианте окружение зависимостей подготовлено заранее, а локальные пакеты
доступны через `PYTHONPATH`; используй указанную команду Python. После работы
выйди из GPU-терминала через `exit`. Изменения Mac не обновляют уже открытую сессию.

## Проверка настройки

Проверено на NVIDIA A100-SXM4-40GB, Python 3.13.3, PyTorch 2.11.0+cu130,
CUDA 13.0, Triton 3.6.0. Контрольная сумма загруженного ядра совпала с локальной:

`09936cf095935d9f9744a9340c2d17e59a60daa1f872e38e5dd4d07dd5f0c2b0`

Результат на проверенной тогда версии кода: `test_flash_forward_pass_triton[False]` прошёл,
`test_flash_forward_pass_triton[True]` упал на сравнении результата с эталоном.
Реализация задания не менялась. Приложение остановилось после завершения тестов.
Результат тестов может измениться после последующих правок ядра.

[Запуск в Modal](https://modal.com/apps/surkovvv/main/ap-Ak4XIMuxAYnHpTWgqUy40w)

[Передача исходников отдельно от образа](https://modal.com/docs/sdk/py/latest/Image#add_local_dir)
