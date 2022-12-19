import sys
import Page1
import Page3
import window
from PyQt5.QtWidgets import QApplication, QMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWnd = QMainWindow()
    ui = window.Ui_MainWindow()
    # 可以理解成将创建的 ui 绑定到新建的 mainWnd 上
    ui.setupUi(mainWnd)
    # page2弃用，为了不引入新bug，故使用隐藏替代删除
    page1 = Page1.Page1(ui, mainWnd)
    page3 = Page3.Page3(ui, mainWnd)
    mainWnd.show()
    sys.exit(app.exec_())
