"""Post-process the generated .docx to match the IEEE template details that
docx-js cannot express: footnotes use symbol marks (*, †, ...) instead of
Arabic numerals, and the title's footnote mark is sized like the title.

Called automatically by build_paper.js; usage: python finalize_docx.py <file.docx>
"""

import re
import shutil
import sys
import tempfile
import zipfile

FOOTNOTE_PR = '<w:footnotePr><w:numFmt w:val="chicago"/></w:footnotePr>'


def main(path):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/document.xml":
                xml = data.decode("utf-8")
                # footnotePr must precede <w:type>/<w:pgSz> inside each sectPr.
                xml, n = re.subn(r"(<w:sectPr(?: [^>]*)?>)(?!<w:footnotePr)", r"\1" + FOOTNOTE_PR, xml)
                if n == 0:
                    sys.exit("no sectPr found")
                xml = xml.replace(
                    '<w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteReference w:id="1"/>',
                    '<w:rPr><w:rStyle w:val="FootnoteReference"/><w:sz w:val="40"/><w:szCs w:val="40"/></w:rPr>'
                    '<w:footnoteReference w:id="1"/>',
                )
                data = xml.encode("utf-8")
            dst.writestr(item, data)
    shutil.move(tmp, path)


if __name__ == "__main__":
    main(sys.argv[1])
