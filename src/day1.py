import platform
import pkg_resources

def main():
    print("Hello, World!")
    print("Python Version :: "+platform.python_version())
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
    for m in installed_packages_list:
        print("Package:-> "+m)

if __name__ == "__main__":
    main()



