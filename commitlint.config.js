/** @type {import('@commitlint/types').UserConfig} */
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // Types permitidos — conventional commits padrão
    'type-enum': [
      2,
      'always',
      [
        'feat',     // nova funcionalidade
        'fix',      // correção de bug
        'docs',     // documentação
        'style',    // formatação (sem mudança de lógica)
        'refactor', // refatoração sem nova feature ou fix
        'perf',     // melhoria de desempenho
        'test',     // adição ou correção de testes
        'build',    // sistema de build, dependências
        'ci',       // configuração de CI/CD
        'chore',    // tarefas de manutenção
        'revert',   // reverter commit anterior
      ],
    ],
    // Scopes opcionais — validação apenas de casing, sem enum fixo
    'scope-case': [2, 'always', 'lower-case'],
    // Cabeçalho
    'header-max-length': [2, 'always', 100],
    'subject-case': [2, 'never', ['sentence-case', 'start-case', 'pascal-case', 'upper-case']],
    'subject-empty': [2, 'never'],
    'subject-full-stop': [2, 'never', '.'],
    // Corpo e rodapé
    'body-leading-blank': [1, 'always'],
    'footer-leading-blank': [1, 'always'],
    'body-max-line-length': [2, 'always', 100],
    'footer-max-line-length': [2, 'always', 100],
  },
};
