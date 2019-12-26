import os
from functools import lru_cache



# better to move to config to void leak
_ip2pwd = {"10.19.66.30": "root!@#.com"}

_address = ["ubuntu@10.19.66.30:/data/turing/turing"]

_source = "/data/zhangkl/turing_new/update-model-1/serving"

_state = """sshpass -p {pwd} scp -r {source} {target}"""


def trans_model(version: int, source: str = None, target: str = None, port: int = None) -> None:
    """
    deploy source/version's model to target/port/version ,port decides ips
    :param version: model version
    :param source: model's path
    :param target: deploy the model to where, add port is the abs_path
    :param port: model category
    :return: None
    """
    @lru_cache(maxsize=20)
    def parse_ip(address):
        ip = address.split(":")[0].split("@")[1]
        return ip

    def parse2list(item):
        if not isinstance(item, (list, tuple)):
            item = [item]
        return item

    source = source or _source
    addresses = parse2list(target or _address.copy())
    # print(addresses)

    for address in addresses:

        real_source = os.path.join(source, str(version))
        real_target = os.path.join(address, str(version))

        # TODO: use thread???
        os.system(_state.format(pwd=_ip2pwd[parse_ip(address)],
                                source=real_source,
                                target=real_target))
        print("trans_model ok")


if __name__ == "__main__":
    trans_model(74)
