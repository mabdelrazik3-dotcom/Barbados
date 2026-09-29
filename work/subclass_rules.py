def subclass_of(f, height_split=150):
    """pixel-only sub-class rule (fitted by work/10_subclass_tree.py). f = line_features(path)."""
    if f['height'] > height_split:
        return _subclass_B(f)
    return _subclass_A(f)


def _subclass_A(f):
    if f['paper'] <= 187.2:
        return 'A01'
    else:
        if f['width'] <= 1048.0:
            if f['width'] <= 941.5:
                return 'A02'
            else:
                if f['stroke_rel'] <= 0.07895:
                    return 'A03'
                else:
                    return 'A04'
        else:
            if f['stroke_rel'] <= 0.06856:
                return 'A05'
            else:
                if f['tint'] <= 49.06:
                    return 'A06'
                else:
                    return 'A07'


def _subclass_B(f):
    if f['width'] <= 5032.0:
        if f['sharp'] <= 0.0987:
            return 'B01'
        else:
            return 'B02'
    else:
        return 'B03'
