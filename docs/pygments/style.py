from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Error, Text, \
     Number, Operator, Generic, Whitespace, Punctuation, Other, Literal


class monokai(Style):
    """
    This style mimics the Monokai color scheme.
    """

    background_color = "#272822"
    highlight_color = "#49483e"

    styles = {
        # No corresponding class for the following:
        Text:                      "#f8f8f2", # class:  '' # noqa
        Whitespace:                "",        # class: 'w'
        Error:                     "#960050 bg:#1e0010", # class: 'err' # noqa
        Other:                     "",        # class 'x'
        ...
    } # noqa