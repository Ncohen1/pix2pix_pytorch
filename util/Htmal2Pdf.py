# from PyQt4.QtGui import QTextDocument, QPrinter, QApplication
#
# import sys
# app = QApplication(sys.argv)
#
# doc = QTextDocument()
# location = "/home/neama_yuval/pytorch-CycleGAN-and-pix2pix/results/ctUs_pix2pix_noDropout/test_latest/index.html"
# html = open(location).read()
# doc.setHtml(html)
#
# printer = QPrinter()
# printer.setOutputFileName("foo.pdf")
# printer.setOutputFormat(QPrinter.PdfFormat)
# printer.setPageSize(QPrinter.A4);
# printer.setPageMargins (15,15,15,15,QPrinter.Millimeter);
#
# doc.print_(printer)
# print("done!")

import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtWebKit import *

app = QApplication(sys.argv)
web = QWebView()
web.load(QUrl("file:///home/neama_yuval/pytorch-CycleGAN-and-pix2pix/checkpoints/ctUs_pix2pix_withDropout/web/index_epoch_2_a.html"))
printer = QPrinter()
printer.setPageSize(printer.A4)
printer.setPageMargins (15,15,15,15,QPrinter.Millimeter);
printer.setOutputFormat(QPrinter.PdfFormat)
printer.setOutputFileName("index_epoch_2_a.pdf")
printer.setFullPage(False)

def convertIt():
    web.print_(printer)
    print("Pdf generated")
    QApplication.exit()

QObject.connect(web, SIGNAL("loadFinished(bool)"), convertIt)
sys.exit(app.exec_())