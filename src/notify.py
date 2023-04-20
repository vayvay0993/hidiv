import win32api
import win32gui
import win32con

message_map = {win32gui.RegisterWindowMessage("TaskbarCreated"): "TaskbarCreated"}


class SystemTrayIcon(object):
    def __init__(self, tooltip):
        self.tooltip = tooltip
        self.visible = False
        message_map.update(
            {win32gui.RegisterWindowMessage("MyCustomMessage"): self.on_custom_message}
        )

    def on_custom_message(self, hwnd, msg, wparam, lparam):
        print("Custom message received!")

    def show(self):
        hinst = win32api.GetModuleHandle(None)
        icon_flags = win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP
        icon_handle = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)
        message = win32gui.RegisterWindowMessage("MyCustomMessage")
        hwnd = win32gui.CreateWindow(
            "BUTTON", "MyWindow", win32con.WS_POPUP, 0, 0, 0, 0, 0, 0, hinst, None
        )
        win32gui.UpdateWindow(hwnd)
        self.visible = True
        nid = (hwnd, 0, icon_flags, message, icon_handle, self.tooltip)
        win32gui.Shell_NotifyIcon(win32gui.NIM_ADD, nid)

    def hide(self):
        if self.visible:
            nid = (self.hwnd, 0)
            win32gui.Shell_NotifyIcon(win32gui.NIM_DELETE, nid)
            self.visible = False


# Create a new instance of the SystemTrayIcon class.
tray_icon = SystemTrayIcon("System Tray Icon Example")

# Show the system tray icon and wait for the user to click on it.
tray_icon.show()
win32gui.PumpMessages()
