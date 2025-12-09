# XXAA

xxaa is a performance analysis utility designed to parse profile logs and generate structured comparison tables. It specifically targets regression testing by comparing current kernel performance against previous releases. This tool provides the objective data necessary to substantiate performance degradation claims when reporting issues to the AI and Compute Library team.

## Usage

To convert profile file, run 

```bash
xxaa convert {profile_file}
```

To compare two profile files, run

```bash
xxaa compare {profile_file_1} {profile_file_2}
```


## Installation

xxaa can be built by [uv](https://docs.astral.sh/uv/)

```bash
uv build 
```

the wheel package is under `dist/` folder.