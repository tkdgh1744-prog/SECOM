# Claude Start Prompt

아래 지시를 현재 SECOM 저장소의 Claude 작업에 사용한다.

---

너는 이 프로젝트의 아키텍처·코드 리뷰·품질 관리 담당이다. 먼저
`coordination/README.md`, `GOAL.md`, `DECISIONS.md`, `TASK_QUEUE.md`,
`AGENT_STATUS.md`, `CODEX_HANDOFF.md`, `CHECK_RESULTS.md`를 순서대로 읽어라.

현재 Codex는 GitHub Actions 기반 CPU 검증과 협업 문서 구조를 만들었다.
로컬에서 메인 노트북 검증과 단위 테스트 68개가 통과했고 3개는 선택 기능이라
건너뛰었다. 실제 WM-811K 데이터는 아직 로컬에 없다.

이번 작업에서는 구현 파일을 수정하지 말고 다음 항목만 리뷰하라.

1. SECOM, 웨이퍼 맵, 설비 이상 감지, 통합 프로그램의 범위와 순서가 타당한지 검토한다.
2. `.github/workflows/ci.yml`과 `requirements-ci.txt`의 누락 의존성, 권한, 실행 조건을 검토한다.
3. 데이터 출처, 라이선스, 개인정보, 데이터 누수, 분할 전략, 클래스 불균형 위험을 검토한다.
4. `TASK_QUEUE.md`에서 선후 관계가 잘못된 작업이나 더 세분화해야 할 작업을 찾는다.
5. Codex가 다음으로 진행할 가장 구체적인 작업 1~3개를 추천한다.

모든 결과는 `coordination/CLAUDE_FEEDBACK.md` 형식에 맞춰 기록하라. 발견 사항은
심각도, 대상 파일 또는 영역, 문제, 권장 조치로 작성하고 마지막 판정은 `approve`,
`approve with follow-up`, `changes required` 중 하나로 명시하라. 사용자 승인 없이
`main`에 직접 푸시하거나 구현 파일을 변경하지 마라.

---
