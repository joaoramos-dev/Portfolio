# João Pedro Ramos — portfólio 3D

Portfólio interativo em React, Vite, Three.js, React Three Fiber e Drei. A cena carrega os modelos em `public/models/`, incluindo a sala e o computador CRT recriado.

## Executar localmente

```bash
npm install
npm run dev
```

Build de produção: `npm run build`; preview do build: `npm run preview`.

## Modelos 3D

Os modelos GLB usados pelo site estão versionados em `public/models/`. O script `scripts/repair_office_model.py` atualiza a sala a partir do arquivo Blender de origem, adicionando flores ao cacto, ajustando as normais da mesa e corrigindo materiais e transparência.

O arquivo `.blend` de origem não está incluído neste repositório. Para executar o script, configure `OFFICE_ASSETS_BLEND` com o caminho local para esse arquivo e use Blender 5.2 ou posterior:

```powershell
$env:OFFICE_ASSETS_BLEND = "C:\caminho\para\OfficeAssets_free.blend"
blender --background $env:OFFICE_ASSETS_BLEND --python scripts\repair_office_model.py -- --blend-output "$env:TEMP\OfficeAssets_portfolio.blend" --glb-output public\models\desk_room_fixed.glb
```

A textura Wood Table 001, do Poly Haven, está disponível sob licença CC0: https://polyhaven.com/a/wood_table_001 · https://polyhaven.com/license

## Interação

A câmera inicial e a iluminação estão em `src/components/DeskScene.jsx`. Clique no CRT, no mural ou na fotografia para aproximar a câmera; use “Voltar à mesa” ou `Esc` para retornar. A cena não possui rotação livre.

A iluminação animada é criada no Three.js; o GLB não contém as luzes da cena. A lâmpada inclui bulbo de vidro, soquete e filamento. O monitor não recebe sombras para evitar artefatos na carcaça.
