from __future__ import annotations

import pathlib
import struct
import typing

from collections import OrderedDict
from dataclasses import dataclass


class CompactValue:
    width = None
    format = '>B'
    transform = lambda x: x  # noqa
    inv_transform = lambda x: x  # noqa
    proxied = False

    def __init__(self, value=None):
        if self.width is None:
            self.width = struct.calcsize(self.format)
        if value is not None:
            self._value = self.__class__.inv_transform(value)

    def to_bytes(self):
        # return self._value.to_bytes(self.width, byteorder='big', signed=False)
        return struct.pack(self.format, self._value)

    def from_bytes(self, bitstream):
        # self._value = int.from_bytes(bitstream[:self.width], byteorder='big', signed=False)
        # return bitstream[self.width:]
        self._value = struct.unpack(self.format, bitstream[:self.width])[0]
        return bitstream[self.width:]

    def to_value(self):
        return self.value

    @property
    def value(self):
        return self.__class__.transform(self._value)

    @value.setter
    def value(self, _v):
        _v = self.__class__.inv_transform(_v)
        bits = struct.pack(self.format, _v)
        new_val = struct.unpack(self.format, bits)[0]
        assert _v == new_val
        self._value = _v

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class CChar(CompactValue):
    format = '>c'
    transform = lambda x: x.decode('utf-8')  # noqa
    inv_transform = lambda x: x.encode('utf-8')  # noqa


class UInt1(CompactValue):
    format = '>B'


class UInt2(CompactValue):
    format = '>H'



# class UInt3(CompactValue):
#     width = 3
#
#     def __init__(self, value=None):
#         super(UInt3, self).__init__(value)

class UInt4(CompactValue):
    format = '>L'


class Int1(CompactValue):
    format = '>b'


class Bool1(CompactValue):
    format = '>?'


class Float2(CompactValue):
    format = '>e'


class Float4(CompactValue):
    format = '>f'


class Float8(CompactValue):
    format = '>d'


class ComplexCompactValue:
    pass


class CompactList(CompactValue, list, ComplexCompactValue):
    value_class = UInt1
    length_type = UInt4

    def __init__(self, values=None):
        super(CompactList, self).__init__(0)

        # self.value_class = value_class
        # print(T.__args__)
        self.length = self.length_type()
        if values is not None:
            for v in values:
                self.append(v)

    def to_bytes(self):
        length = super().__len__()
        # self.value = length
        self.length.value = length
        # bitstream = super(CompactList, self).to_bytes()
        bitstream = self.length.to_bytes()
        for obj in self:
            bitstream += obj.to_bytes()
        return bitstream

    def from_bytes(self, bitstream):
        self.clear()
        # new_bitstream = super(CompactList, self).from_bytes(bitstream)
        new_bitstream = self.length.from_bytes(bitstream)
        for i in range(self.length.value):
            new_value = self.value_class()
            new_bitstream = new_value.from_bytes(new_bitstream)
            self.append(new_value)
        return new_bitstream

    def append(self, __object) -> None:
        # print('list', __object)
        if type(__object) is self.value_class:
            super(CompactList, self).append(__object)
        elif isinstance(__object, CompactValue):
            raise ValueError('Error')
        else:
            super(CompactList, self).append(self.value_class(__object))

    def __getitem__(self, item):
        value: CompactValue = super().__getitem__(item)
        if not isinstance(value, ComplexCompactValue):
            return value.to_value()

    def to_value(self):
        result = []
        for c_v in self:
            result.append(c_v.to_value())
        return result

    def __repr__(self):
        s_el = [str(obj) for obj in self]
        return f"{self.__class__.__name__}[{', '.join(s_el)}]"


class UInt1List(CompactList):
    value_class = UInt1
    proxied = True


class UInt2List(CompactList):
    value_class = UInt2
    proxied = True


class Float4List(CompactList):
    value_class = Float4


class CharList(CompactList):
    value_class = CChar


class String2(CharList):
    length_type = UInt2
    proxied = True

    @property
    def value(self):
        return self.to_value()

    @value.setter
    def value(self, _v):
        for c in _v:
            self.append(c)

    def to_value(self):
        result = []
        for c_v in self:
            result.append(c_v.to_value())
        return ''.join(result)


# class UInt3List(CompactList):
#     def __init__(self, values=None):
#         super(UInt3List, self).__init__(UInt3, values)


class CompactDict(CompactValue, dict, ComplexCompactValue):
    def __init__(self, dict_classes: OrderedDict[str, type]):
        super().__init__(0)
        self.__dict_classes = dict_classes

    def __getitem__(self, item):
        assert item in self.__dict_classes.keys()
        return super().__getitem__(item)

    def has_key(self, key):
        return key in self.__dict_classes.keys()

    def __setitem__(self, key, value):
        # print('call setitem in CompactDict', 'k', key, 'v', value)
        try:
            value_cls = self.__dict_classes[key]
        except KeyError:
            raise KeyError(f'Unknown field `{key}`')
        # print('call setitem in CompactDict', value_cls)
        if isinstance(value, (CompactDict, CompactList)):
            super().__setitem__(key, value)
        elif isinstance(value, CompactValue):
            raise ValueError(f'Use bare value for field {key}')
        else:
            super().__setitem__(key, value_cls(value))
            # print('before setattr')
        super().__setattr__(key, value)

    def to_bytes(self):
        bitstream = b''
        for k, v in self.__dict_classes.items():
            val = self[k]
            bitstream += val.to_bytes()
        length = len(bitstream)
        self.value = length
        bitstream = super().to_bytes() + bitstream
        return bitstream

    def from_bytes(self, bitstream):
        self.clear()
        new_bitstream = super().from_bytes(bitstream)
        for k, val_cls in self.__dict_classes.items():
            try:
                new_value = val_cls()
                new_bitstream = new_value.from_bytes(new_bitstream)

                # print('before dict set item', k)
                super().__setitem__(k, new_value)
                if isinstance(new_value, (CompactDict, CompactList)):
                    super().__setattr__(k, new_value)
                else:
                    super().__setattr__(k, new_value.to_value())
                # print('after set')

            except Exception as e:

                e.args = tuple([e.args[0], f'key `{k}: {val_cls}`'])
                raise e
        return new_bitstream

    def to_value(self):
        result = OrderedDict()
        for k, v in self.__dict_classes.items():
            # print('k', k, 'v',  v)
            val = self[k]
            result[k] = val.to_value()
        return result


class CompactClass(CompactDict):

    def __getattribute__(self, item):
        obj = super().__getattribute__(item)
        try:
            if obj.proxied:
                return obj.to_value()
            return obj
        except AttributeError:
            return obj

    def __setattr__(self, key, value):
        # print('setattr', key, value)
        try:
            super().__setitem__(key, value)
            super().__setattr__(key, value)
        except AttributeError:
            super().__setattr__(key, value)
        except KeyError:
            super().__setattr__(key, value)


@dataclass
class _hint:  # Helper class for pycharm type hint # noqa
    pass


class CompactDataclass(CompactClass, _hint):
    def __init__(self):
        reversed_mro = list(reversed(self.__class__.mro()))
        # idx = 0
        idx = reversed_mro.index(CompactDataclass)
        initialized_values = {}
        definitions = {}
        for i in range(idx + 1, len(reversed_mro)):
            cls = reversed_mro[i]
            # print(cls)
            new_definitions = cls.__annotations__
            # print(cls, cls.__dict__)

            definitions.update(new_definitions)

            for k, v in new_definitions.items():
                # print(k, v)
                try:
                    initialized_values[k] = cls.__dict__[k]
                except KeyError:
                    pass

        super(CompactClass, self).__init__(typing.cast(OrderedDict[str, type],  definitions))

        for k, v in initialized_values.items():
            self[k] = v

    def load_dict(self, data):
        for k, v in data.items():
            self[k] = v



# Type alies
uint1 = UInt1
uint2 = UInt2
uint4 = UInt4
int1 = Int1
# uint3 = UInt3
bool1 = Bool1
float4 = Float4
char1 = CChar
clist = CompactList
string2 = String2
