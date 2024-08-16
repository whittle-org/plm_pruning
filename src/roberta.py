from transformers.models.roberta.modeling_roberta import (
    RobertaForSequenceClassification,
    RobertaForMultipleChoice,
)

from model_wrapper.mask import mask_roberta
from search_spaces import (
    SmallSearchSpace,
    LayerSearchSpace,
    FullSearchSpace,
    MediumSearchSpace,
)


class ROBERTASuperNetMixin:
    search_space = None
    handles = None

    def select_sub_network(self, sub_network_config):
        head_mask, ffn_mask = self.search_space.config_to_mask(sub_network_config)
        head_mask = head_mask.to(device="cuda", dtype=self.dtype)
        ffn_mask = ffn_mask.to(device="cuda", dtype=self.dtype)
        self.handles = mask_roberta(self.roberta, ffn_mask, head_mask)

    def reset_super_network(self):
        for handle in self.handles:
            handle.remove()


class ROBERTASuperNetMixinLAYERSpace(ROBERTASuperNetMixin):
    @property
    def search_space(self):
        return LayerSearchSpace(self.config)


class ROBERTASuperNetMixinMEDIUMSpace(ROBERTASuperNetMixin):
    @property
    def search_space(self):
        return MediumSearchSpace(self.config)


class ROBERTASuperNetMixinLARGESpace(ROBERTASuperNetMixin):
    @property
    def search_space(self):
        return FullSearchSpace(self.config)


class ROBERTASuperNetMixinSMALLSpace(ROBERTASuperNetMixin):
    @property
    def search_space(self):
        return SmallSearchSpace(self.config)


class SuperNetRobertaForSequenceClassificationSMALL(
    RobertaForSequenceClassification, ROBERTASuperNetMixinSMALLSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetRobertaForMultipleChoiceSMALL(
    RobertaForMultipleChoice, ROBERTASuperNetMixinSMALLSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetRobertaForSequenceClassificationLAYER(
    RobertaForSequenceClassification, ROBERTASuperNetMixinLAYERSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetRobertaForMultipleChoiceLAYER(
    RobertaForMultipleChoice, ROBERTASuperNetMixinLAYERSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetRobertaForSequenceClassificationMEDIUM(
    RobertaForSequenceClassification, ROBERTASuperNetMixinMEDIUMSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetRobertaForMultipleChoiceMEDIUM(
    RobertaForMultipleChoice, ROBERTASuperNetMixinMEDIUMSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetRobertaForSequenceClassificationLARGE(
    RobertaForSequenceClassification, ROBERTASuperNetMixinLARGESpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetRobertaForMultipleChoiceLARGE(
    RobertaForMultipleChoice, ROBERTASuperNetMixinLARGESpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)
