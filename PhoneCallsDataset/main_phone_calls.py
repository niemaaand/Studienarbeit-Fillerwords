from PhoneCallsDataset import prepreprocessing
import create_master_csv


def main():
    call_home = prepreprocessing.CallHomeIntoStructure("C:/CallHomeDataset/")
    #call_home.bring_to_structure()
    #call_home.update_master_csv()


if __name__ == '__main__':
    main()