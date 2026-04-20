from glob import glob
import os

from setuptools import setup


package_name = "lc_web_control"


setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*launch.[pxy][yma]*")),
        (os.path.join("share", package_name, "static"), glob("lc_web_control/static/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="cmls",
    maintainer_email="cmls@todo.todo",
    description="Web dashboard for controlling SLAM, map preview, and vision tracking.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "web_control = lc_web_control.web_control_node:main",
        ],
    },
)
