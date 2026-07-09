# Python 绑定状态与扩展建议

当前 Python 绑定目录中没有独立的 `cqlib.error_mitigation` 运行时模块。因此官方文档应明确：

- Rust 核心层已有错误缓解实现；
- Python 用户暂时不能直接通过 `import cqlib.error_mitigation` 使用；
- 若后续补齐绑定，应增加正式教程和 API 参考。

## 建议的 Python API 形态

```python
from cqlib.error_mitigation import ZNEMitigation, ExtrapolateMethod

zne = ZNEMitigation(circuit, fold_levels=[0, 1, 2])
folded = zne.fold_circuits()
```

## 建议的教程

1. ZNE：Bell 态或 1 比特旋转线路；
2. ZNE：选择性门折叠；
3. Virtual Distillation：2 copies 示例；
4. 与 `DensityMatrixNoise` 的联动；
5. 与真实后端 estimator 的联动。

<!-- expanded_by_chatgpt_20260617 -->
## ：发布前必须确认的入口

```python
# 在正式写入 Python 用户教程前，请实际验证：
# import cqlib.error_mitigation
# from cqlib.error_mitigation import ZNEMitigation, VirtualDistillation
```

如果导入失败，应在教程中保留 Rust Core 说明，并把 Python 示例标注为未来接口草案，避免用户复制后直接报错。
